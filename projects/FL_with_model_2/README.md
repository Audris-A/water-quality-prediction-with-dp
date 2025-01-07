# Flower FL with DP for water quality prediction

## Developed using the Flower example for TensorFlow/Keras

### Dependencies

Install `tensorflow`, `tensorflow_privacy` and `flwr` packages.

### Run Federated Learning

Launch the server: 
```shell
python3 server.py
```

Now start the four clients with partitions from 0 to 3 and dp-stages from 0 to 2 (0 - for eps < 1, 1 - for eps < 5, 2 - for eps < 10). One client example:
```shell
python3 client.py --partition-id 0 --dp-stage 0
```
