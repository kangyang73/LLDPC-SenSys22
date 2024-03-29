# LLDPC-SenSys22 Codes

Created by [Kang Yang](https://www.kangyangg.com).

The GNN network is built based on the [Factor-Graph-Neural-Network](https://github.com/zzhang1987/Factor-Graph-Neural-Network).

## Requirements
The following packages are required: 

1. Python 3 
2. PyTorch 1.13

### Dataset downloading

Download the sample dataset from [dataset](https://www.kangyangg.com/data/lldpc_sensys22.pt).

### Folder Structure
```shell
--root_folder
    ---lldpc_code
        --codes
        --lib
        --utils
        --README.md
        --configure.json
        --main_run.py
    ---data_log
        --data
            --sf7
                --snr1
                    data.pt
    ---output
```

### Running

Before running the code, in the beginning of the `if __name__ == '__main__':` section, you need to set the `base_dir` variable to your `root_folder` path.

```shell
python main_run.py
```


