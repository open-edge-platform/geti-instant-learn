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
        # Strip query strings and fragments
        zipfile_clean="${zipfile%%\?*}"
        zipfile_clean="${zipfile_clean%%#*}"

        # Validate no path separators or traversal
        if [[ "$zipfile_clean" == */* ]] || [[ "$zipfile_clean" == *..* ]]; then
            echo "Warning: Skipping unsafe filename: $zipfile"
            continue
        fi

        # Validate proper .zip extension and not empty
        if [[ ! "$zipfile_clean" =~ ^[^/]+\.zip$ ]] || [ "$zipfile_clean" = ".zip" ]; then
            echo "Warning: Skipping invalid filename: $zipfile"
            continue
        fi

        dataset_name="${zipfile_clean%.zip}"
        dataset_subdir="$DATASET_DIR/$dataset_name"
        archive_path="$DATASET_DIR/$zipfile_clean"

        if [ -d "$dataset_subdir" ] && [ "$(ls -A "$dataset_subdir")" ]; then
            echo "Dataset $dataset_name already exists and is not empty. Skipping."
            continue
        fi

        mkdir -p "$dataset_subdir"
        echo "Downloading $zipfile_clean from ${BASE_URL}${zipfile}"

        if ! curl -fsSL -o "$archive_path" "${BASE_URL}${zipfile}"; then
            echo "Error: Failed to download $zipfile"
            continue
        fi

        echo "Extracting $zipfile_clean to $dataset_subdir"
        unzip -j -q -o "$archive_path" -d "$dataset_subdir"

        echo "Removing archive $zipfile_clean"
        rm "$archive_path"
        echo ""
    done

# Test the download-datasets recipe with malicious filenames (dry-run mode)
test-dataset-security:
    #!/usr/bin/env bash
    echo "Testing dataset download security..."
    echo ""

    # Create test HTML with malicious filenames
    TEST_HTML="/tmp/test_malicious_listing.html"
    cat > "$TEST_HTML" << 'EOF'
    <!DOCTYPE html>
    <html><body><ul>
    <li><a href=safe.zip>safe.zip</a></li>
    <li><a href=another-safe_file.zip>another-safe_file.zip</a></li>
    <li><a href=../../etc/passwd.zip>../../etc/passwd.zip</a></li>
    <li><a href=../../../tmp/evil.zip>../../../tmp/evil.zip</a></li>
    <li><a href=subdir/file.zip>subdir/file.zip</a></li>
    <li><a href=file.zip?token=abc>file.zip?token=abc</a></li>
    <li><a href=file.zip#anchor>file.zip#anchor</a></li>
    <li><a href=data.zip?query=1#fragment>data.zip?query=1#fragment</a></li>
    <li><a href=notazip.txt>notazip.txt</a></li>
    <li><a href=..zip>..zip</a></li>
    <li><a href=.zip>.zip</a></li>
    <li><a href=/absolute/path.zip>/absolute/path.zip</a></li>
    </ul></body></html>
    EOF

    # Simulate the parsing logic
    echo "Parsing test HTML..."
    zip_files=$(cat "$TEST_HTML" | grep -oE 'href=[^>]+\.zip' | sed 's/href=//' | sort -u)

    echo "Found entries:"
    echo "$zip_files"
    echo ""

    echo "Testing validation logic:"
    for zipfile in $zip_files; do
        # Strip query strings and fragments
        zipfile_clean="${zipfile%%\?*}"
        zipfile_clean="${zipfile_clean%%#*}"

        # Validate no path separators or traversal
        if [[ "$zipfile_clean" == */* ]] || [[ "$zipfile_clean" == *..* ]]; then
            echo "✓ BLOCKED (path traversal): $zipfile"
            continue
        fi

        # Validate proper .zip extension and not empty
        if [[ ! "$zipfile_clean" =~ ^[^/]+\.zip$ ]] || [ "$zipfile_clean" = ".zip" ]; then
            echo "✓ BLOCKED (invalid name): $zipfile"
            continue
        fi

        echo "✓ ALLOWED: $zipfile → $zipfile_clean"
    done

    rm "$TEST_HTML"
    echo ""
    echo "Security test completed!"
