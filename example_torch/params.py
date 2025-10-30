from cuvis_ai.anomoly.rx_v2 import RXPerBatch
from cuvis_ai.normalization.normalization import MinMaxNormalizer
            

# {'param_a': 10, 'param_2': 11, 'param_3': 'value'}
model = RXPerBatch(eps=1e-99)
print(model.hparams)  # Output: {'eps': 1e-99}

model = MinMaxNormalizer(eps=1e-99)
print(model.hparams)  # Output: {'eps': 1e-99}

print(model.serialize(""))
