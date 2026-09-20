import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import numpy as np

model = keras.models.load_model('models/recovered_model.h5', compile=False)

print('=' * 70)
print('DEEP DIAGNOSTIC AUDIT - recovered_model.h5')
print('=' * 70)

# ======== AUDIT 1: Architecture ========
print('\n[AUDIT 1] ARCHITECTURE')
print('-' * 50)
print('Total layers:', len(model.layers))
print('Input shape:', model.input_shape)
last_layer = model.layers[-1]
print('Final layer:', last_layer.name, '| Type:', last_layer.__class__.__name__)
cfg = last_layer.get_config()
print('Final activation:', cfg.get('activation', 'unknown'))
print('Output units:', cfg.get('units', 'unknown'))

# ======== AUDIT 2: Trainable vs Frozen ========
print('\n[AUDIT 2] TRAINABLE vs FROZEN')
print('-' * 50)
total_params = model.count_params()
trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
print('Total params:', f'{total_params:,}')
print('Trainable params:', f'{trainable_params:,}', f'({trainable_params/total_params*100:.2f}%)')
print('Non-trainable (frozen):', f'{non_trainable_params:,}', f'({non_trainable_params/total_params*100:.2f}%)')

# ======== AUDIT 3: Weight Distribution ========
print('\n[AUDIT 3] WEIGHT HEALTH (Last 3 weighted layers)')
print('-' * 50)
weighted_layers = [l for l in model.layers if len(l.get_weights()) > 0]
for layer in weighted_layers[-3:]:
    weights = layer.get_weights()
    w = weights[0]
    print('  Layer:', layer.name, '| shape:', w.shape)
    print('    mean:', round(np.mean(w), 6), '| std:', round(np.std(w), 6))
    print('    min:', round(float(np.min(w)), 6), '| max:', round(float(np.max(w)), 6))
    near_zero = np.mean(np.abs(w) < 0.01) * 100
    large = np.mean(np.abs(w) > 1.0) * 100
    print('    near-zero(<0.01):', round(near_zero, 1), '% | large(>1.0):', round(large, 1), '%')
    if large > 20:
        print('    [WARNING] High percentage of large weights')
    print()

# ======== AUDIT 4: Dead Neurons ========
print('[AUDIT 4] DEAD NEURON CHECK')
print('-' * 50)
dense_layer = None
for l in model.layers:
    if 'dense_2' in l.name:
        dense_layer = l
        break
if dense_layer:
    dw = dense_layer.get_weights()
    if len(dw) > 1:
        biases = dw[1]
        dead = int(np.sum(biases < -5.0))
        print('Dense-256 biases: min=', round(float(np.min(biases)), 4),
              'max=', round(float(np.max(biases)), 4),
              'mean=', round(float(np.mean(biases)), 4))
        print('Dead neurons (bias < -5):', dead, '/', len(biases))
        if dead > 128:
            print('[RED FLAG] >50% dead neurons')
        else:
            print('[PASS] Neuron health acceptable')

# ======== AUDIT 5: Smoke Test ========
print('\n[AUDIT 5] PREDICTION SMOKE TEST')
print('-' * 50)

black = np.zeros((1, 224, 224, 3), dtype='float32')
pred_black = float(model.predict(black, verbose=0).flatten()[0])
print('Pure BLACK:', round(pred_black, 6))

white = np.ones((1, 224, 224, 3), dtype='float32')
pred_white = float(model.predict(white, verbose=0).flatten()[0])
print('Pure WHITE:', round(pred_white, 6))

noise = np.random.rand(1, 224, 224, 3).astype('float32')
pred_noise = float(model.predict(noise, verbose=0).flatten()[0])
print('Random NOISE:', round(pred_noise, 6))

gap = abs(pred_black - pred_white)
if gap < 0.05:
    print('[RED FLAG] Model cannot distinguish black from white')
else:
    print('[PASS] Black-White discrimination gap:', round(gap, 4))

# ======== AUDIT 6: Variance Test ========
print('\n[AUDIT 6] OUTPUT VARIANCE TEST (100 random images)')
print('-' * 50)
batch = np.random.rand(100, 224, 224, 3).astype('float32')
preds = model.predict(batch, verbose=0).flatten()
print('Mean:', round(float(np.mean(preds)), 4))
print('Std:', round(float(np.std(preds)), 4))
print('Min:', round(float(np.min(preds)), 4), '| Max:', round(float(np.max(preds)), 4))
print('< 0.3 (Healthy):', np.sum(preds < 0.3), '/100')
print('0.3-0.7 (Uncertain):', np.sum((preds >= 0.3) & (preds <= 0.7)), '/100')
print('> 0.7 (RP):', np.sum(preds > 0.7), '/100')

if np.std(preds) < 0.02:
    print('[RED FLAG] Model collapsed - nearly identical outputs')
elif np.mean(preds) > 0.85:
    print('[WARNING] Model has positive bias (tends to predict RP)')
elif np.mean(preds) < 0.15:
    print('[WARNING] Model has negative bias (tends to predict Healthy)')
else:
    print('[PASS] Output distribution looks healthy')

# ======== AUDIT 7: BatchNorm Statistics ========
print('\n[AUDIT 7] BATCHNORM STATISTICS (training convergence)')
print('-' * 50)
bn_layers = [l for l in model.layers if 'bn' in l.name.lower() or 'batch' in l.name.lower()]
if bn_layers:
    last_bn = bn_layers[-1]
    bn_weights = last_bn.get_weights()
    if len(bn_weights) >= 4:
        gamma, beta, moving_mean, moving_var = bn_weights
        print('Last BN layer:', last_bn.name)
        print('  Moving mean range: [', round(float(np.min(moving_mean)), 4), ',', round(float(np.max(moving_mean)), 4), ']')
        print('  Moving var range: [', round(float(np.min(moving_var)), 4), ',', round(float(np.max(moving_var)), 4), ']')
        if np.max(moving_var) < 0.001:
            print('[RED FLAG] Variance near zero - possible training collapse')
        else:
            print('[PASS] BatchNorm statistics appear healthy')

print('\n' + '=' * 70)
print('AUDIT COMPLETE')
print('=' * 70)
