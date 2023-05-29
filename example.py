from leurn import LEURN, load_data, plot_explaination, read_partition_process_data, train_model
import os
if not os.path.exists('tmp'):
   os.makedirs('tmp')
data = load_data("housing")
X_train, X_val, X_test, y_train, y_val, y_test, y_max, X_names, X_mean, X_std = read_partition_process_data(
    data, target_name="median_house_value", task_type="reg"
)
model: LEURN = train_model(X_train, y_train, X_val, y_val, task_type="reg", output_path="tmp", epoch_no=100)
test_sample=X_test[0:1,:]
explain = model.explain(test_sample, feat_names=X_names, y_max=y_max)
plot_explaination(explain, "tmp/explain.png")