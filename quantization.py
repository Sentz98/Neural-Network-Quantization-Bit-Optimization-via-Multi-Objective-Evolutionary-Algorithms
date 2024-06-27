from collections import namedtuple
import torch
import torch.nn.functional as F

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val,num_bits):
  # Calc Scale and zero point of next 
  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale
  
  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)

  return scale, zero_point

def quantize_tensor(x, num_bits, min_val=None, max_val=None, symmetric=False):
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x =  x / scale
    if not symmetric:
      q_x = q_x + zero_point
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x, symmetric=False):
    if symmetric:
      return  q_x.scale * q_x.tensor.float()
    else:
      return q_x.scale * (q_x.tensor.float() - q_x.zero_point)
    
def quantizeLayer(x, layer, stat, scale_x, zp_x, num_bits):
  # cache old values
  W = layer.weight.data
  B = layer.bias.data

  # quantise weights (activations are already quantised)
  # TODO: find if there is a way to pre quantize them
  w = quantize_tensor(layer.weight.data, num_bits)
  b = quantize_tensor(layer.bias.data, num_bits)

  layer.weight.data = w.tensor.float()
  layer.bias.data = b.tensor.float()

  # This is Quantisation Artihmetic
  scale_w = w.scale
  zp_w = w.zero_point
  scale_b = b.scale
  zp_b = b.zero_point

  scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'], num_bits=num_bits)

  # Preparing input by shifting
  X = x.float() - zp_x
  layer.weight.data = scale_x * scale_w*(layer.weight.data - zp_w)
  layer.bias.data = scale_b*(layer.bias.data + zp_b)

  # All int computation
  x = (layer(X)/ scale_next) + zero_point_next

  # Reset weights for next forward pass
  layer.weight.data = W
  layer.bias.data = B

  return x, scale_next, zero_point_next

def QLinearLayer(x, layer, stat, relu=False):
  #assert x is a QTensor
  assert type(x) == QTensor, "Input to quantizeLayer should be a QTensor"
 
  W = layer.weight.data
  B = layer.bias.data
  # quantise weights (activations are already quantised) 
  x_hat = dequantize_tensor(x)

  if isinstance(stat['bits'], list):
    assert len(stat['bits']) == W.size(0), "Number of bits for each neuron should be equal to number of neurons in the layer"
    w_hat = torch.zeros_like(W)
    b_hat = torch.zeros_like(B)
    for i, nbit in enumerate(stat['bits']):
      #prendi iesima riga di W e iesimo elemento di B e quantizza il tensore
      bi = B[i].unsqueeze(0)
      neuron = torch.cat((W[i], bi))
      neuron_hat = dequantize_tensor(quantize_tensor(neuron, nbit)) 
      #split weights and bias (the last element is the bias)
      w_hat[i] = neuron_hat[:-1]
      b_hat[i] = neuron_hat[-1]    
  else:
    w_hat = dequantize_tensor(quantize_tensor(W, stat['bits']))
    b_hat = dequantize_tensor(quantize_tensor(B, stat['bits']))

  layer.weight.data = w_hat
  layer.bias.data = b_hat
  y = layer(x_hat)

  if relu:
    y = F.relu(y)

  if isinstance(stat['bits'], list):
    y = quantize_tensor(y, stat['bits_out'], min_val=stat['min'], max_val=stat['max'])
  else:
    y = quantize_tensor(y, stat['bits'], min_val=stat['min'], max_val=stat['max'])

  # Reset weights for next forward pass
  layer.weight.data = W
  layer.bias.data = B
  
  return y

# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
  max_val, _ = torch.max(x, dim=1)
  min_val, _ = torch.min(x, dim=1)
  
  if key not in stats:
    stats[key] = {"max": max_val.max(), "min": min_val.min()}
  else:
    stats[key]['max'] = torch.max(stats[key]['max'], max_val.max())
    stats[key]['min'] = torch.min(stats[key]['min'], min_val.min())
  
  return stats

# Reworked Forward Pass to access activation Stats through updateStats function

def MLPActivationStats(model, x, stats):

  x = x.view(-1, 28*28)
    
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'input')

  x = model.fc1(x)

  x = model.relu(x)

  stats = updateStats(x, stats, 'fc1')

  x = model.fc2(x)

  stats = updateStats(x, stats, 'fc2')

  return stats

# Entry function to get stats of all functions.
def gatherStats(model, test_loader, device):
    model.eval()
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = MLPActivationStats(model, data, stats)
    
    final_stats = {}
    
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"], "min" : value["min"]}

    final_stats['fc1']['#params'] = model.fc1.weight.numel() + model.fc1.bias.numel()
    final_stats['fc2']['#params'] = model.fc2.weight.numel() + model.fc2.bias.numel()

    return final_stats

def model_size(stats):
    total_size = 0
    for _, value in stats.items():
        # if in value there is a #params key, then it is a weight layer
        if '#params' in value:
            if isinstance(value['bits'], list):
              neur_params = value['#params']/len(value['bits'])
              total_size += sum([b*neur_params for b in value['bits']])
            else:
              total_size += value['#params']*value['bits']
    return total_size

