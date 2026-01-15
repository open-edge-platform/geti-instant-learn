
template-dataset-url := "https://storage.geti.intel.com/geti-prompt/Frames.zip"

check-proxy:
    #!/usr/bin/env bash
    if [ -z ${https_proxy+x} ]; then
        echo "Error: https_proxy is unset";
    else
        echo "https_proxy is set to '$https_proxy'";
    fi

download-dataset target_dir: check-proxy
    #!/usr/bin/env bash
    DATASET_DIR="{{ target_dir }}"

    if [ -d "$DATASET_DIR" ] && [ "$(ls -A $DATASET_DIR)" ]; then
        echo "Dataset directory $DATASET_DIR already exists and is not empty. Skipping download."
        exit 0
    fi
    mkdir -p $DATASET_DIR
    echo "Downloading default dataset from {{ template-dataset-url }}"
    if ! curl -s {{ template-dataset-url }} -o Frames.zip; then
        echo "Error: Failed to download dataset from {{ template-dataset-url }}"
        exit 0  # proceed without dataset
    fi
    echo "Unpacking default dataset to $DATASET_DIR"
    unzip -j -q -o Frames.zip -d $DATASET_DIR
    echo "Removing downloaded dataset archive Frames.zip"
    rm Frames*.zip
