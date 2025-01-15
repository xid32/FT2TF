import os
import numpy as np



def check_metrics(files_path, metric, mode):
    steps_list = os.listdir(files_path)
    metric_list = []
    valid_step_list = []
    for step in steps_list:
        step_path = os.path.join(files_path, step)
        txt_path = os.path.join(step_path, "test_results.txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = list(f)
            for content_i in content:
                if metric in content_i:
                    content_i = float(content_i.strip().replace(metric + " : ", ""))
                    metric_list.append(content_i)
                    valid_step_list.append(step_path)
                else:
                    continue
    valid_step = np.array(valid_step_list)
    metric_all = np.array(metric_list)
    if mode == "max":
        max_idx = np.argmax(metric_list)
        metric = metric_all[max_idx]
        step = valid_step[max_idx]
    else:
        min_idx = np.argmin(metric_list)
        metric = metric_all[min_idx]
        step = valid_step[min_idx]

    txt_path = os.path.join(step, "test_results.txt")
    print(txt_path)
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = list(f)
        for content_i in content:
            content_i = content_i.strip()
            print(content_i)
    return

if __name__ == '__main__':
    files_path = "/jumbo/jinlab/XD/face_expression/project/ckpts/ckpt_lrs3_1019"
    metric = "FID"
    mode = "min"
    check_metrics(files_path, metric, mode)






