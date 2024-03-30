
for x,y,y_pred in zip(denorm_x_test,denorm_y_test,denorm_predictions):
    y_index = y.transpose()[0].index(max(y.transpose()[0]))