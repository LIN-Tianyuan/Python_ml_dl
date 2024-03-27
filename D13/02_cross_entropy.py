"""
cross-entropy
"""
import math

# Sample categories: 3      5 categories in total
y_true = [0, 0, 0, 1, 0]    # 100%
pred_y = [0.1, 0.1, 0.1, 0.6, 0.1]  # 60%
pred_y1 = [0.1, 0.1, 0.05, 0.7, 0.05]   # 70%
pred_y2 = [0.1, 0.03, 0.02, 0.8, 0.05]  # 80%
pred_y3 = [0.02, 0.02, 0.03, 0.9, 0.03]  # 90%

entropy1 = 0.0
entropy2 = 0.0
entropy3 = 0.0
entropy4 = 0.0
for i in range(len(y_true)):
    entropy1 += y_true[i] * math.log(pred_y[i])
    entropy2 += y_true[i] * math.log(pred_y1[i])
    entropy3 += y_true[i] * math.log(pred_y2[i])
    entropy4 += y_true[i] * math.log(pred_y3[i])

print('Cross-entropy1:', -entropy1)
print('Cross-entropy2:', -entropy2)
print('Cross-entropy3:', -entropy3)
print('Cross-entropy4:', -entropy4)

"""
total_sample = 60000
batch_size = 100    # Batch size
total_batch = int(total_sample / batch_size)

epoch = 10  # round

for i in range(epoch):  # 10 rounds of training (60,000)
    for j in range(total_batch):
        # Get a batch and train.
"""