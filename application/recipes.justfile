dataset-base-url := "https://storage.geti.intel.com/instant-learn/datasets/"

# Dynamically discover and download all zip files from the dataset URL
download-datasets target_dir:
    #!/usr/bin/env bash
    BASE_URL="{{ dataset-base-url }}"
    DATASET_DIR="{{ target_dir }}"

    mkdir -p "$DATASET_DIR"

    echo "Fetching dataset listing from $BASE_URL"

    # Fetch directory listing and extract .zip file names
    listing=$(curl -fsSL "$BASE_URL")
    if [ $? -ne 0 ]; then
        echo "Error: Failed to fetch dataset listing from $BASE_URL"
        echo "Check network connection, SSL certificates, or URL accessibility"
        exit 1
    fi

    zip_files=$(echo "$listing" | grep -oE 'href=[^>]+\.zip' | sed 's/href=//' | sort -u)

    if [ -z "$zip_files" ]; then
        echo "No zip files found at $BASE_URL"
        exit 1
    fi

    echo "Found zip files:"
    echo "$zip_files"
    echo ""

    # Download and extract each zip file
    for zipfile in $zip_files; do
        dataset_name="${zipfile%.zip}"
        dataset_subdir="$DATASET_DIR/$dataset_name"

        if [ -d "$dataset_subdir" ] && [ "$(ls -A "$dataset_subdir")" ]; then
            echo "Dataset $dataset_name already exists and is not empty. Skipping."
            continue
        fi

        mkdir -p "$dataset_subdir"
        echo "Downloading $zipfile from ${BASE_URL}${zipfile}"

        if ! curl -fsSL -o "$zipfile" "${BASE_URL}${zipfile}"; then
            echo "Error: Failed to download $zipfile"
            continue
        fi

        echo "Extracting $zipfile to $dataset_subdir"
        unzip -j -q -o "$zipfile" -d "$dataset_subdir"

        echo "Removing archive $zipfile"
        rm "$zipfile"
        echo ""
    done
