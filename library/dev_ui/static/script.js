/*
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

document.addEventListener("DOMContentLoaded", () => {
  const processButton = document.getElementById("runButton");
  const pipelineSelect = document.getElementById("pipelineSelect");
  const samNameSelect = document.getElementById("samNameSelect");
  const encoderModelSelect = document.getElementById("encoderModelSelect");
  const datasetSelect = document.getElementById("datasetSelect");
  const classNameSelect = document.getElementById("classNameSelect");
  const nShotInput = document.getElementById("nShotInput");
  const numBackgroundPointsInput = document.getElementById(
    "numBackgroundPointsInput",
  );
  const similarityThresholdInput = document.getElementById(
    "similarityThresholdInput",
  );
  const maskSimilarityThresholdInput = document.getElementById(
    "maskSimilarityThresholdInput",
  );
  const numTargetImagesControl = document.getElementById("numTargetImagesControl");
  const numTargetImagesInput = document.getElementById("numTargetImagesInput");
  const numTargetImagesValueSpan = document.getElementById("numTargetImagesValue");
  const maxTargetImagesValueSpan = document.getElementById("maxTargetImagesValue");
  const resultsContainer = document.getElementById("results-container");
  const progressContainer = document.getElementById("progress-container");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const progressText = document.getElementById("progress-text");
  const MAX_CANVAS_WIDTH = 500;
  const canvasDataStore = {};
  const maskImageCache = {};
  const groundTruthMaskImageCache = {};
  const similarityThresholdValueSpan = document.getElementById("similarityThresholdValue");

  const maskSimilarityThresholdValueSpan = document.getElementById(
    "maskSimilarityThresholdValue",
  );

  const referenceContainer = document.getElementById("reference-container");

  const randomPriorCheckbox = document.getElementById("randomPriorCheckbox");
  const compileModelsCheckbox = document.getElementById("compileModelsCheckbox");
  const precisionSelect = document.getElementById("precisionSelect");

  async function fetchAndPopulateClasses(selectedDataset) {
    classNameSelect.innerHTML =
      '<option value="" disabled selected>Loading...</option>';
    numTargetImagesControl.classList.remove("hidden");
    numTargetImagesValueSpan.textContent = "N/A";
    maxTargetImagesValueSpan.textContent = "";
    numTargetImagesInput.disabled = true;

    try {
      const response = await fetch(
        `/api/classes?dataset=${encodeURIComponent(selectedDataset)}`,
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || `HTTP error! status: ${response.status}`,
        );
      }
      const data = await response.json();
      classNameSelect.innerHTML =
        '<option value="" disabled selected>Select Class...</option>';
      if (data.classes && data.classes.length > 0) {
        const sortedClasses = data.classes.sort();

        classNameSelect.innerHTML =
          '<option value="" disabled>Select Class...</option>';
        sortedClasses.forEach((className) => {
          const option = document.createElement("option");
          option.value = className;
          option.textContent = className;
          classNameSelect.appendChild(option);
        });
        // Pre-select the first actual class (index 1, since index 0 is the disabled placeholder)
        if (classNameSelect.options.length > 1) {
          classNameSelect.selectedIndex = 1;
          // Trigger update for the pre-selected class
          updateTargetImageSlider(selectedDataset, classNameSelect.value);
        } else {
          numTargetImagesValueSpan.textContent = "N/A";
          maxTargetImagesValueSpan.textContent = "N/A";
          numTargetImagesInput.disabled = true;
        }
      } else {
        classNameSelect.innerHTML =
          '<option value="" disabled selected>No classes found</option>';
        if (!classNameSelect.value) {
          numTargetImagesValueSpan.textContent = "N/A";
          maxTargetImagesValueSpan.textContent = "N/A";
          numTargetImagesInput.disabled = true;
        }
      }
    } catch (error) {
      console.error("Error fetching classes:", error);
      classNameSelect.innerHTML =
        '<option value="" disabled selected>Error loading</option>';
      if (!classNameSelect.value) {
        numTargetImagesValueSpan.textContent = "N/A";
        maxTargetImagesValueSpan.textContent = "N/A";
        numTargetImagesInput.disabled = true;
      }
    }
  }

  // Event listener for dataset change
  datasetSelect.addEventListener("change", () => {
    console.log("Dataset select changed!");
    numTargetImagesControl.classList.remove("hidden");
    numTargetImagesValueSpan.textContent = "N/A";
    maxTargetImagesValueSpan.textContent = "N/A";
    numTargetImagesInput.disabled = true;
    fetchAndPopulateClasses(datasetSelect.value);
  });

  // Event listener for class name change
  classNameSelect.addEventListener("change", () => {
    console.log("Class select changed!");
    updateTargetImageSlider(datasetSelect.value, classNameSelect.value);
  });

  // Event listener for N-Shot change
  nShotInput.addEventListener("change", () => {
    console.log("N-Shot input changed!");
    updateTargetImageSlider(datasetSelect.value, classNameSelect.value);
  });

  // Initial population
  fetchAndPopulateClasses(datasetSelect.value);

  if (similarityThresholdInput && similarityThresholdValueSpan) {
    similarityThresholdInput.addEventListener('input', (event) => {
      similarityThresholdValueSpan.textContent = event.target.value;
    });
  }

  if (maskSimilarityThresholdInput && maskSimilarityThresholdValueSpan) {
    maskSimilarityThresholdInput.addEventListener('input', (event) => {
      maskSimilarityThresholdValueSpan.textContent = event.target.value;
    });
    maskSimilarityThresholdValueSpan.textContent = maskSimilarityThresholdInput.value;
  } else {
    console.error("Mask similarity threshold slider or value span not found!");
  }

  async function updateTargetImageSlider(selectedDataset, selectedClassName) {
    numTargetImagesControl.classList.remove("hidden");
    numTargetImagesValueSpan.textContent = "Loading...";
    maxTargetImagesValueSpan.textContent = "";
    numTargetImagesInput.disabled = true;

    if (!selectedDataset || !selectedClassName) {
      // If no dataset/class selected (e.g., after dataset change), show N/A
      numTargetImagesValueSpan.textContent = "N/A";
      maxTargetImagesValueSpan.textContent = "N/A";
      // Keep it disabled
      return;
    }

    try {
      const response = await fetch(
        `/api/class_info?dataset=${encodeURIComponent(selectedDataset)}&class_name=${encodeURIComponent(selectedClassName)}`,
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      const totalImages = data.total_images || 0;
      const nShot = parseInt(nShotInput.value, 10) || 1;
      const maxTargets = Math.max(1, totalImages - nShot); // Ensure at least 1

      numTargetImagesInput.max = maxTargets;
      // Set value to max by default, or keep existing if valid
      let currentValue = parseInt(numTargetImagesInput.value, 10);
      if (isNaN(currentValue) || currentValue > maxTargets || currentValue < 1) {
        currentValue = maxTargets;
      }
      numTargetImagesInput.value = currentValue;
      numTargetImagesValueSpan.textContent = currentValue;
      maxTargetImagesValueSpan.textContent = maxTargets;
      numTargetImagesControl.classList.remove("hidden"); // Already visible, but good practice
      numTargetImagesInput.disabled = false; // Enable input
    } catch (error) {
      console.error("Error fetching class info:", error);
      // Keep control visible but show error state
      numTargetImagesValueSpan.textContent = "Error";
      maxTargetImagesValueSpan.textContent = "N/A";
      numTargetImagesInput.disabled = true;
    }
  }

  if (numTargetImagesInput && numTargetImagesValueSpan) {
    numTargetImagesInput.addEventListener('input', (event) => {
      numTargetImagesValueSpan.textContent = event.target.value;
    });
  } else {
    console.error("Target image slider or value span not found!");
  }

  processButton.addEventListener("click", async () => {
    const pipeline = pipelineSelect.value;
    const samName = samNameSelect.value;
    const encoderModel = encoderModelSelect.value;
    const dataset = datasetSelect.value;
    const className = classNameSelect.value;
    const nShot = parseInt(nShotInput.value, 10);
    const numTargetImages = parseInt(numTargetImagesInput.value, 10);
    const numBackgroundPoints = parseInt(numBackgroundPointsInput.value, 10);
    const similarityThreshold = parseFloat(similarityThresholdInput.value);
    const maskSimilarityThreshold = parseFloat(maskSimilarityThresholdInput.value);
    const useRandomPrior = randomPriorCheckbox?.checked ?? false;
    const compileModels = compileModelsCheckbox?.checked ?? true;
    const precision = precisionSelect.value;

    if (!className || isNaN(nShot) || nShot < 1 || isNaN(similarityThreshold) || isNaN(maskSimilarityThreshold) || isNaN(numTargetImages) || numTargetImages < 1) {
      // Use progress text for errors before starting
      progressContainer.classList.remove("hidden");
      progressText.textContent = "Invalid input.";
      progressBarFill.style.width = "0%";
      progressBarFill.classList.add("bg-red-600"); // Make bar red on error
      progressBarFill.classList.remove("bg-blue-600");
      return;
    }

    // --- UI Update for Loading State ---
    const originalButtonText = processButton.innerHTML;
    processButton.disabled = true;
    processButton.innerHTML = `
            <svg class="animate-spin -ml-1 mr-2 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Processing...
        `;
    resultsContainer.innerHTML = "";
    progressContainer.classList.remove("hidden");
    progressBarFill.style.width = "0%";
    progressBarFill.classList.remove("bg-red-600");
    progressBarFill.classList.add("bg-blue-600");
    progressText.textContent = useRandomPrior ? "Loading pipeline with Random Prior..." : "Loading pipeline...";
    Object.keys(canvasDataStore).forEach((key) => delete canvasDataStore[key]);
    Object.keys(maskImageCache).forEach((key) => delete maskImageCache[key]);
    Object.keys(groundTruthMaskImageCache).forEach((key) => delete groundTruthMaskImageCache[key]);
    if (referenceContainer) referenceContainer.innerHTML = '';

    // --- Fetch and Process Streamed Response ---
    let totalTargets = 0;
    let resultsCount = 0;
    let errorOccurred = false;

    try {
      const response = await fetch("/api/process", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pipeline: pipeline,
          dataset: dataset,
          class_name: className,
          n_shot: nShot,
          num_target_images: numTargetImages,
          num_background_points: numBackgroundPoints,
          sam: samName,
          encoder_model: encoderModel,
          similarity_threshold: similarityThreshold,
          mask_similarity_threshold: maskSimilarityThreshold,
          random_prior: useRandomPrior,
          compile_models: compileModels,
          precision: precision,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      // --- Stream Reading Logic ---
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let isFirstChunk = true; // Flag to handle the first message
      console.log("Starting stream reading loop...");

      while (true) {
        console.log("Waiting for reader.read()...");
        const { done, value } = await reader.read();
        console.log(`Read chunk: done=${done}, value size=${value?.length}`);

        if (done) {
          console.log("Stream finished.");
          break;
        }

        const decodedChunk = decoder.decode(value, { stream: true });
        console.log("Decoded chunk:", decodedChunk);
        buffer += decodedChunk;
        console.log("Current buffer:", buffer);

        let newlineIndex;
        while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
          const line = buffer.substring(0, newlineIndex).trim();
          buffer = buffer.substring(newlineIndex + 1);
          console.log("Processing line:", line);

          if (line) {
            try {
              const dataChunk = JSON.parse(line);
              console.log("Parsed data chunk:", dataChunk);

              if (isFirstChunk) {
                // The first chunk contains total_targets and reference_data
                if (dataChunk.total_targets !== undefined) {
                  totalTargets = parseInt(dataChunk.total_targets, 10) || 0;
                  console.log("Received total targets:", totalTargets);
                  progressText.textContent = `Processing 0 / ${totalTargets}...`;
                }
                if (dataChunk.reference_data) {
                  console.log("Received reference data:", dataChunk.reference_data);
                  displayReferenceData(dataChunk.reference_data);
                }
                isFirstChunk = false; // Move to processing results/errors after first chunk
              } else if (dataChunk.error) {
                console.error("Backend processing error:", dataChunk.error);
                progressText.textContent = `Error: ${dataChunk.error}`;
                progressBarFill.classList.remove("bg-blue-600");
                progressBarFill.classList.add("bg-red-600");
                errorOccurred = true;
              } else if (dataChunk.target_results) {
                if (errorOccurred) continue;

                const receivedCount = dataChunk.target_results.length;
                resultsCount += receivedCount;
                console.log(`Calling displayResults for ${receivedCount} items.`);
                displayResults(dataChunk.target_results); // Append results incrementally

                // Update progress bar
                if (totalTargets > 0) {
                  const percentage = Math.min(100, (resultsCount / totalTargets) * 100);
                  progressBarFill.style.width = `${percentage}%`;
                  progressText.textContent = `Processing ${resultsCount} / ${totalTargets}...`;
                } else {
                  progressText.textContent = `Received ${resultsCount} results... (Total unknown)`;
                }
              }
            } catch (e) {
              console.error("Error parsing JSON chunk:", e, "Chunk:", line);
              progressText.textContent = `Error parsing chunk: ${line.substring(0, 50)}...`;
              progressBarFill.classList.remove("bg-blue-600");
              progressBarFill.classList.add("bg-red-600");
              errorOccurred = true;
              break;
            }
          }
        }
        if (errorOccurred) break;
      }
      // --- End Stream Reading Logic ---

      if (!errorOccurred) {
        if (resultsCount === 0 && totalTargets === 0 && !isFirstChunk) {
          progressText.textContent = "Processing complete. No results returned.";
        } else if (resultsCount === totalTargets) {
          progressBarFill.style.width = `100%`;
          progressText.textContent = `Processing complete (${resultsCount} / ${totalTargets}).`;
        } else if (resultsCount < totalTargets) {
          progressText.textContent = `Processing incomplete (${resultsCount} / ${totalTargets}). Stream ended unexpectedly.`;
          progressBarFill.classList.remove("bg-blue-600");
          progressBarFill.classList.add("bg-yellow-500");
        }
      }

    } catch (error) {
      console.error("Error fetching or processing data:", error);
      progressContainer.classList.remove("hidden");
      progressText.textContent = `Error: ${error.message}`;
      progressBarFill.style.width = "100%";
      progressBarFill.classList.remove("bg-blue-600");
      progressBarFill.classList.add("bg-red-600");
      errorOccurred = true;
    } finally {
      console.log("Executing finally block...");
      processButton.disabled = false;
      console.log(`Run button disabled state: ${processButton.disabled}`);
      processButton.innerHTML = originalButtonText;
    }
  });

  function displayReferenceData(referenceData) {
    if (!referenceContainer) return;
    referenceContainer.innerHTML = ''; // Clear previous content

    if (!referenceData || referenceData.length === 0) {
      referenceContainer.innerHTML = '<p class="text-sm text-gray-500">No reference samples provided.</p>';
      return;
    }

    const header = document.createElement('h3');
    header.classList.add('text-xl', 'font-semibold', 'mb-3', 'text-gray-800');
    header.textContent = `Reference Samples (N=${referenceData.length}):`;
    referenceContainer.appendChild(header);

    const gridContainer = document.createElement('div');
    gridContainer.classList.add('grid', 'grid-cols-[repeat(auto-fill,minmax(150px,1fr))]', 'gap-4'); // Responsive grid

    referenceData.forEach((refItem, index) => {
      const itemDiv = document.createElement('div');
      itemDiv.classList.add('reference-item', 'relative', 'border', 'border-gray-200', 'rounded-md', 'overflow-hidden', 'shadow-sm');

      const img = document.createElement('img');
      img.src = refItem.image_data_uri;
      img.alt = `Reference Image ${index + 1}`;
      img.classList.add('block', 'w-full', 'h-auto');
      itemDiv.appendChild(img);

      if (refItem.mask_data_uri) {
        const maskImg = document.createElement('img');
        maskImg.src = refItem.mask_data_uri;
        maskImg.alt = `Reference Mask ${index + 1}`;
        maskImg.classList.add('absolute', 'top-0', 'left-0', 'w-full', 'h-full', 'opacity-50', 'pointer-events-none'); // Overlay style
        itemDiv.appendChild(maskImg);
      }

      const label = document.createElement('span');
      label.textContent = `Ref ${index + 1}`;
      label.classList.add('absolute', 'bottom-1', 'right-1', 'bg-black', 'bg-opacity-50', 'text-white', 'text-xs', 'px-1.5', 'py-0.5', 'rounded');
      itemDiv.appendChild(label);

      gridContainer.appendChild(itemDiv);
    });

    referenceContainer.appendChild(gridContainer);
  }

  function displayResults(targetResults) {
    targetResults.forEach((result, indexOffset) => {
      // Calculate a unique index based on how many results are already displayed
      const existingResultsCount = resultsContainer.children.length;
      const uniqueIndex = existingResultsCount + indexOffset;

      const targetItemDiv = document.createElement("div");
      targetItemDiv.classList.add(
        "target-item",
        "bg-white",
        "p-4",
        "rounded-lg",
        "shadow",
        "flex",
        "flex-col",
        "gap-4",
      );
      targetItemDiv.id = `target-item-${uniqueIndex}`;

      const canvasContainer = document.createElement("div");
      canvasContainer.classList.add("flex-shrink-0");

      const canvasId = `canvas-${uniqueIndex}`;
      const canvas = document.createElement("canvas");
      canvas.id = canvasId;
      canvas.classList.add(
        "border",
        "border-gray-300",
        "rounded",
        "max-w-full",
      );
      canvasContainer.appendChild(canvas);

      const controlsContainer = document.createElement("div");
      controlsContainer.classList.add(
        "item-controls",
        "flex-grow",
        "space-y-4",
      );

      // --- Point Display Mode Grouped Buttons ---
      const pointModeHeader = document.createElement("strong");
      pointModeHeader.classList.add(
        "block",
        "text-sm",
        "font-medium",
        "text-gray-700",
        "mb-1",
      );
      pointModeHeader.textContent = "Point Display Mode:";
      controlsContainer.appendChild(pointModeHeader);

      const pointModeGroup = document.createElement("div");
      pointModeGroup.classList.add("inline-flex", "rounded-md", "shadow-sm");
      pointModeGroup.setAttribute("role", "group");
      const pointModeGroupId = `point-mode-group-${canvasId}`;
      pointModeGroup.id = pointModeGroupId;

      // Base classes for buttons
      const buttonBaseClasses = [
        "relative",
        "inline-flex",
        "items-center",
        "px-3",
        "py-2",
        "text-sm",
        "font-semibold",
        "ring-1",
        "ring-inset",
        "ring-gray-300",
        "focus:z-10",
      ];
      const buttonSelectedClasses = [
        "bg-primary",
        "text-white",
        "hover:bg-blue-700",
      ];
      const buttonUnselectedClasses = [
        "bg-white",
        "text-gray-900",
        "hover:bg-gray-50",
      ];

      // Used Points Button
      const buttonUsed = document.createElement("button");
      buttonUsed.type = "button";
      buttonUsed.textContent = "Used Points";
      buttonUsed.dataset.pointMode = "used";
      buttonUsed.classList.add(
        ...buttonBaseClasses,
        "rounded-l-md",
        ...buttonSelectedClasses,
      );

      // All Points Button
      const buttonAll = document.createElement("button");
      buttonAll.type = "button";
      buttonAll.textContent = "All Points";
      buttonAll.dataset.pointMode = "all";
      buttonAll.classList.add(
        ...buttonBaseClasses,
        "-ml-px",
        "rounded-r-md",
        ...buttonUnselectedClasses,
      );

      pointModeGroup.appendChild(buttonUsed);
      pointModeGroup.appendChild(buttonAll);
      controlsContainer.appendChild(pointModeGroup);

      // Event listener for the group
      pointModeGroup.addEventListener("click", (event) => {
        console.log("Point mode group clicked!");
        const clickedButton = event.target.closest("button");
        if (!clickedButton || !pointModeGroup.contains(clickedButton)) return;

        const newMode = clickedButton.dataset.pointMode;
        const currentData = canvasDataStore[canvasId];
        if (!currentData) return;
        const currentMode = currentData.pointMode;

        if (newMode !== currentMode) {
          currentData.pointMode = newMode;

          const buttons = pointModeGroup.querySelectorAll("button");
          buttons.forEach((button) => {
            button.classList.remove(
              ...buttonSelectedClasses,
              ...buttonUnselectedClasses,
            );
            if (button.dataset.pointMode === newMode) {
              button.classList.add(...buttonSelectedClasses);
            } else {
              button.classList.add(...buttonUnselectedClasses);
            }
          });

          redrawCanvas(canvasId);
        }
      });

      // --- Ground Truth Toggle ---
      const gtToggleDiv = document.createElement("div");
      gtToggleDiv.classList.add("gt-toggle-control", "mt-4", "flex", "items-center");

      const gtCheckbox = document.createElement("input");
      gtCheckbox.type = "checkbox";
      gtCheckbox.id = `gt-toggle-${canvasId}`;
      gtCheckbox.dataset.canvasId = canvasId;
      gtCheckbox.classList.add(
        "h-4",
        "w-4",
        "text-green-600",
        "border-gray-300",
        "rounded",
        "focus:ring-green-500",
      );
      gtCheckbox.checked = false;

      const gtLabel = document.createElement("label");
      gtLabel.htmlFor = gtCheckbox.id;
      gtLabel.textContent = "Show Ground Truth Mask (Green)";
      gtLabel.classList.add("ml-2", "block", "text-sm", "text-gray-700");

      gtToggleDiv.appendChild(gtCheckbox);
      gtToggleDiv.appendChild(gtLabel);
      controlsContainer.appendChild(gtToggleDiv);

      // Event listener for GT toggle
      gtCheckbox.addEventListener("change", (event) => {
        const targetCanvasId = event.target.dataset.canvasId;
        if (canvasDataStore[targetCanvasId]) {
          canvasDataStore[targetCanvasId].showGroundTruth = event.target.checked;
          redrawCanvas(targetCanvasId);
        }
      });
      // --- End Ground Truth Toggle ---

      const maskControlsDiv = document.createElement("div");
      maskControlsDiv.classList.add("mask-controls", "space-y-2", "mt-4");
      maskControlsDiv.innerHTML =
        '<strong class="block text-sm font-medium text-gray-700 mb-1">Masks:</strong>';

      const maskButtonContainer = document.createElement("div");
      maskButtonContainer.classList.add("flex", "gap-x-2", "mb-2");

      // Function to create styled buttons
      const createButton = (text) => {
        const button = document.createElement("button");
        button.textContent = text;
        button.classList.add(
          "px-2.5",
          "py-1.5",
          "border",
          "border-gray-300",
          "rounded-md",
          "shadow-sm",
          "text-xs",
          "font-medium",
          "text-gray-700",
          "bg-white",
          "hover:bg-gray-50",
          "focus:outline-none",
          "focus:ring-2",
          "focus:ring-offset-2",
          "focus:ring-indigo-500",
        );
        button.dataset.canvasId = canvasId;
        return button;
      };

      const selectAllButton = createButton("Select All");
      selectAllButton.addEventListener("click", (event) => {
        const targetCanvasId = event.target.dataset.canvasId;
        const maskCheckboxesContainer = document.getElementById(
          `mask-checkboxes-${targetCanvasId}`,
        );
        if (maskCheckboxesContainer) {
          const checkboxes = maskCheckboxesContainer.querySelectorAll(
            'input[type="checkbox"]',
          );
          checkboxes.forEach((cb) => (cb.checked = true));
          redrawCanvas(targetCanvasId);
        }
      });

      const unselectAllButton = createButton("Unselect All");
      unselectAllButton.addEventListener("click", (event) => {
        const targetCanvasId = event.target.dataset.canvasId;
        const maskCheckboxesContainer = document.getElementById(
          `mask-checkboxes-${targetCanvasId}`,
        );
        if (maskCheckboxesContainer) {
          const checkboxes = maskCheckboxesContainer.querySelectorAll(
            'input[type="checkbox"]',
          );
          checkboxes.forEach((cb) => (cb.checked = false));
          redrawCanvas(targetCanvasId);
        }
      });

      maskButtonContainer.appendChild(selectAllButton);
      maskButtonContainer.appendChild(unselectAllButton);
      maskControlsDiv.appendChild(maskButtonContainer);

      const maskCheckboxesContainer = document.createElement("div");
      maskCheckboxesContainer.id = `mask-checkboxes-${canvasId}`;
      maskCheckboxesContainer.classList.add(
        "space-y-1",
        "overflow-y-auto",
        "max-h-40",
      );
      maskControlsDiv.appendChild(maskCheckboxesContainer);

      controlsContainer.appendChild(maskControlsDiv);

      targetItemDiv.appendChild(canvasContainer);
      targetItemDiv.appendChild(controlsContainer);
      resultsContainer.appendChild(targetItemDiv);

      // Store data needed for redraws - use unique canvasId
      const augmentedMasks = (result.masks || []).map(mask => ({
        ...mask,
        isCheckboxHovered: false,
        isPointHovered: false,
      }));

      canvasDataStore[canvasId] = {
        image: null,
        masks: augmentedMasks,
        points: {
          used: result.used_points || [],
          all: (result.prior_points || []).concat(result.used_points || [])
        },
        element: canvas,
        clickablePoints: [],
        pointMode: "used",
        scaleX: 1,
        scaleY: 1,
        originalWidth: 0,
        originalHeight: 0,
        similarityMaps: result.similarity_maps || [],
        ground_truth_mask_uri: result.gt_mask_uri || null, // Store GT Mask URI
        showGroundTruth: false, // State for GT visibility
        currentlyPointHoveredMaskIndex: null, // For point hover tracking
      };

      // Similarity Map Display
      console.log(
        `[Canvas ${canvasId}] Checking for similarity maps:`,
        result.similarity_maps,
      );
      const simMaps = canvasDataStore[canvasId].similarityMaps;
      if (simMaps && simMaps.length > 0) {
        console.log(
          `[Canvas ${canvasId}] Found ${simMaps.length} similarity map(s). Creating container.`,
        );
        const simMapContainer = document.createElement("div");
        simMapContainer.classList.add("similarity-map-container");
        simMapContainer.innerHTML =
          '<h5 class="text-lg font-medium text-gray-900 mb-2">Similarity Maps:</h5>';

        const simMapGrid = document.createElement("div");
        simMapGrid.classList.add(
          "grid",
          "grid-cols-2",
          "sm:grid-cols-3",
          "md:grid-cols-4",
          "lg:grid-cols-5",
          "gap-2",
        );
        simMaps.forEach((mapData, mapIndex) => {
          const mapDiv = document.createElement("div");

          const mapImg = document.createElement("img");
          mapImg.src = mapData.map_data_uri;
          mapImg.alt = `Similarity Map ${mapIndex + 1} (Point ${mapData.point_index})`;
          mapImg.classList.add(
            "w-full",
            "h-auto",
            "border",
            "border-gray-300",
            "rounded",
          );
          mapImg.dataset.canvasId = canvasId;
          mapImg.dataset.mapUri = mapData.map_data_uri;

          const uriLength = mapData.map_data_uri
            ? mapData.map_data_uri.length
            : 0;

          mapDiv.appendChild(mapImg);
          simMapGrid.appendChild(mapDiv);
        });

        simMapContainer.appendChild(simMapGrid);
        const controlsContainer = targetItemDiv.querySelector('.item-controls');
        controlsContainer.appendChild(simMapContainer);
      }
      canvas.addEventListener("click", handleCanvasClick);

      // Add mousemove listener for point hovering
      canvas.addEventListener("mousemove", (event) => {
        const currentData = canvasDataStore[canvasId];
        if (!currentData || !currentData.masks) return;

        const rect = canvas.getBoundingClientRect();
        // Calculate scaled mouse coordinates, similar to handleCanvasClick
        const scaleMouseX = canvas.width / rect.width;
        const scaleMouseY = canvas.height / rect.height;
        const x = (event.clientX - rect.left) * scaleMouseX;
        const y = (event.clientY - rect.top) * scaleMouseY;
        let aPointWasHovered = false;
        let newHoveredMaskIndex = null;

        for (const pPoint of currentData.clickablePoints) {
          const distance = Math.sqrt((x - pPoint.x) ** 2 + (y - pPoint.y) ** 2);
          if (distance < pPoint.radius) {
            newHoveredMaskIndex = pPoint.maskIndex;
            aPointWasHovered = true;
            break;
          }
        }

        if (currentData.currentlyPointHoveredMaskIndex !== newHoveredMaskIndex) {
          // Clear previous highlight if any
          if (currentData.currentlyPointHoveredMaskIndex !== null && currentData.masks[currentData.currentlyPointHoveredMaskIndex]) {
            currentData.masks[currentData.currentlyPointHoveredMaskIndex].isPointHovered = false;
          }
          // Set new highlight if any
          if (newHoveredMaskIndex !== null && currentData.masks[newHoveredMaskIndex]) {
            currentData.masks[newHoveredMaskIndex].isPointHovered = true;
          }
          currentData.currentlyPointHoveredMaskIndex = newHoveredMaskIndex;
          redrawCanvas(canvasId);
        }
      });

      // Add mouseout listener for canvas to clear point hover
      canvas.addEventListener("mouseout", () => {
        const currentData = canvasDataStore[canvasId];
        if (!currentData || !currentData.masks) return;

        if (currentData.currentlyPointHoveredMaskIndex !== null && currentData.masks[currentData.currentlyPointHoveredMaskIndex]) {
          currentData.masks[currentData.currentlyPointHoveredMaskIndex].isPointHovered = false;
          currentData.currentlyPointHoveredMaskIndex = null;
          redrawCanvas(canvasId);
        }
      });

      // Create checkboxes and add listeners
      if (result.masks && result.masks.length > 0) {
        result.masks.forEach((_, maskIndex) => { // Iterate with original maskIndex
          const mask = canvasDataStore[canvasId].masks[maskIndex]; // Get augmented mask
          const checkboxId = `${canvasId}-mask-${mask.instance_id}`;

          // Create container for checkbox + label for styling
          const maskDiv = document.createElement("div");
          maskDiv.classList.add("flex", "items-center");

          const label = document.createElement("label");
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.id = checkboxId;
          checkbox.value = mask.instance_id;
          checkbox.checked = true;
          checkbox.dataset.canvasId = canvasId;
          checkbox.dataset.maskIndex = maskIndex; // Set maskIndex for the event listener
          checkbox.classList.add(
            "h-4",
            "w-4",
            "text-indigo-600",
            "border-gray-300",
            "rounded",
            "focus:ring-indigo-500",
          );

          label.htmlFor = checkboxId;
          label.textContent = `Mask ${maskIndex + 1}`;
          label.classList.add("ml-2", "block", "text-sm");

          maskDiv.appendChild(checkbox);
          maskDiv.appendChild(label);
          maskCheckboxesContainer.appendChild(maskDiv);

          // Add listener to redraw when checkbox changes
          checkbox.addEventListener("change", (event) => {
            const targetCanvasId = event.target.dataset.canvasId;
            redrawCanvas(targetCanvasId);
          });

          // Add hover listeners to the maskDiv for better UX
          maskDiv.addEventListener("mouseenter", () => {
            const currentMask = canvasDataStore[canvasId].masks[maskIndex];
            if (currentMask) {
              currentMask.isCheckboxHovered = true;
              redrawCanvas(canvasId);
            }
          });

          maskDiv.addEventListener("mouseleave", () => {
            const currentMask = canvasDataStore[canvasId].masks[maskIndex];
            if (currentMask) {
              currentMask.isCheckboxHovered = false;
              redrawCanvas(canvasId);
            }
          });
        });
      } else {
        maskCheckboxesContainer.innerHTML += "<i>No masks found</i>";
      }

      preloadMaskImages(canvasId);
      preloadGroundTruthMaskImage(canvasId);

      const img = new Image();
      img.onload = () => {
        const naturalWidth = img.naturalWidth;
        const naturalHeight = img.naturalHeight;
        canvasDataStore[canvasId].originalWidth = naturalWidth;
        canvasDataStore[canvasId].originalHeight = naturalHeight;

        let targetWidth = naturalWidth;
        let targetHeight = naturalHeight;

        if (naturalWidth > MAX_CANVAS_WIDTH) {
          const scaleRatio = MAX_CANVAS_WIDTH / naturalWidth;
          targetWidth = MAX_CANVAS_WIDTH;
          targetHeight = naturalHeight * scaleRatio;
        }

        canvas.width = targetWidth;
        canvas.height = targetHeight;

        canvasDataStore[canvasId].scaleX = targetWidth / naturalWidth;
        canvasDataStore[canvasId].scaleY = targetHeight / naturalHeight;

        canvasDataStore[canvasId].image = img;
        redrawCanvas(canvasId);
      };
      img.onerror = () => {
        console.error(`Failed to load image from data URI`);
        const ctx = canvas.getContext("2d");
        canvas.width = 300;
        canvas.height = 100;
        ctx.fillStyle = "red";
        ctx.font = "16px sans-serif";
        ctx.fillText("Error loading image", 10, 50);
      };
      img.src = result.image_data_uri;
    });
  }

  function preloadMaskImages(canvasId) {
    const data = canvasDataStore[canvasId];
    if (!data || !data.masks) return;

    data.masks.forEach((mask) => {
      const uri = mask.mask_data_uri;
      if (uri && !maskImageCache[uri]) {
        const maskImg = new Image();
        maskImg.onload = () => {
          maskImageCache[uri] = maskImg;
        };
        maskImg.onerror = () => {
          console.error(`Failed to preload mask image from data URI`);
          maskImageCache[uri] = null;
        };
        maskImageCache[uri] = "loading";
        maskImg.src = uri;
      }
    });
  }

  function preloadGroundTruthMaskImage(canvasId) {
    const data = canvasDataStore[canvasId];
    if (!data || !data.ground_truth_mask_uri) return;

    const uri = data.ground_truth_mask_uri;
    if (uri && !groundTruthMaskImageCache[uri]) {
      const gtMaskImg = new Image();
      gtMaskImg.onload = () => {
        groundTruthMaskImageCache[uri] = gtMaskImg;
        console.log(`[Canvas ${canvasId}] Preloaded GT mask.`);
      };
      gtMaskImg.onerror = () => {
        console.error(`[Canvas ${canvasId}] Failed to preload GT mask image from data URI`);
        groundTruthMaskImageCache[uri] = null; // Mark as failed
      };
      groundTruthMaskImageCache[uri] = "loading"; // Mark as loading
      gtMaskImg.src = uri;
    }
  }

  function redrawCanvas(canvasId) {
    const data = canvasDataStore[canvasId];
    if (!data || !data.image) return;

    const canvas = data.element;
    const ctx = canvas.getContext("2d");
    const img = data.image;
    const scaleX = data.scaleX || 1;
    const scaleY = data.scaleY || 1;
    const baseSize = Math.max(3, Math.min(canvas.width, canvas.height) * 0.01);
    const squareSize = baseSize * 1.5;
    const pointRadius = baseSize * 1.2;
    const clickRadius = pointRadius * 1.5;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const maskCheckboxesContainer = document.getElementById(
      `mask-checkboxes-${canvasId}`,
    );
    const visibleMaskIds = new Set();
    if (maskCheckboxesContainer) {
      const checkboxes = maskCheckboxesContainer.querySelectorAll(
        'input[type="checkbox"]:checked',
      );
      checkboxes.forEach((cb) => visibleMaskIds.add(cb.value));
    } else {
      console.warn(
        `Could not find mask checkboxes container for canvas: ${canvasId}`,
      );
    }

    data.masks.forEach((mask) => {
      if (visibleMaskIds.has(mask.instance_id)) {
        const maskImg = maskImageCache[mask.mask_data_uri];
        if (maskImg && maskImg !== 'loading' && maskImg.complete) {
          const shouldHighlight = mask.isCheckboxHovered || mask.isPointHovered;
          if (shouldHighlight) {
            ctx.save();
            // Apply a distinct highlight, e.g., brightness, contrast, and a white drop shadow
            ctx.filter = 'brightness(1.4) contrast(1.1) drop-shadow(0 0 4px #FFFFFF)';
          }

          ctx.globalAlpha = 0.5; // Original opacity for mask
          ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
          ctx.globalAlpha = 1.0;

          if (shouldHighlight) {
            ctx.restore(); // Restore context to remove filter for subsequent drawings
          }
        } else if (!maskImg || maskImg === 'loading') {
        } else {
          console.error(`Failed mask image encountered: ${mask.instance_id}`);
        }
      }
    });

    if (data.showGroundTruth && data.ground_truth_mask_uri) {
      const gtMaskImg = groundTruthMaskImageCache[data.ground_truth_mask_uri];
      if (gtMaskImg && gtMaskImg !== 'loading' && gtMaskImg.complete) {
        ctx.globalAlpha = 0.6;
        ctx.drawImage(gtMaskImg, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
      } else {
        console.error(`[Canvas ${canvasId}] Failed GT mask image encountered.`);
      }
    }

    data.clickablePoints = [];

    // Define point colors and styles (Restore original definitions)
    const OUTLINE_COLOR = "rgba(255, 255, 255, 0.8)";
    const OUTLINE_WIDTH = 1.5;
    const usedForegroundPointColor = "rgba(50, 205, 50, 1)"; // Lime Green for Used Foreground
    const allForegroundPointColor = "rgba(135, 206, 250, 1)"; // Light Sky Blue for All Foreground
    const backgroundPointColor = "rgba(255, 0, 0, 1)"; // Red for Background

    // Draw points based on the current toggle state
    const showAllPoints = data.pointMode === "all";
    const allPoints = (data.points && data.points.all) ? data.points.all : [];
    const usedPoints = (data.points && data.points.used) ? data.points.used : [];
    const pointsToDraw = showAllPoints ? allPoints : usedPoints;
    let foregroundPointIndex = -1; // Index relative to only foreground points in usedPoints

    // First, draw all points from the selected list (either all or used)
    pointsToDraw.forEach((point) => {
      const drawX = point.x * scaleX;
      const drawY = point.y * scaleY;
      const label = point.label;

      if (usedPoints.includes(point) && label === 1) {
        foregroundPointIndex++;
      }

      if (label === 0) {
        ctx.fillStyle = backgroundPointColor;
        ctx.strokeStyle = OUTLINE_COLOR;
        ctx.lineWidth = OUTLINE_WIDTH;
        ctx.beginPath();
        ctx.rect(
          drawX - squareSize / 2,
          drawY - squareSize / 2,
          squareSize,
          squareSize,
        );
        ctx.fill();
        ctx.stroke();
      } else {
        // Foreground points (label > 0)
        let fillColor;
        if (showAllPoints) {
          // Showing all points -> use light blue
          fillColor = allForegroundPointColor;
        } else {
          // Showing only used points -> use green
          fillColor = usedForegroundPointColor;
        }

        // Restore: Draw Foreground points as filled circles with white outline
        ctx.fillStyle = fillColor;
        ctx.strokeStyle = OUTLINE_COLOR;
        ctx.lineWidth = OUTLINE_WIDTH;
        ctx.beginPath();
        ctx.arc(drawX, drawY, pointRadius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }

      if (label === 1 && usedPoints.includes(point)) {
        data.clickablePoints.push({
          x: drawX,
          y: drawY,
          radius: clickRadius,
          maskIndex: foregroundPointIndex
        });
      }
    });

    if (showAllPoints) {
      foregroundPointIndex = -1;
      usedPoints.forEach((point) => {
        if (point.label === 1) {
          foregroundPointIndex++; // Increment here as well
          const drawX = point.x * scaleX;
          const drawY = point.y * scaleY;

          ctx.fillStyle = usedForegroundPointColor;
          ctx.strokeStyle = OUTLINE_COLOR;
          ctx.lineWidth = OUTLINE_WIDTH;
          ctx.beginPath();
          ctx.arc(drawX, drawY, pointRadius, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();

          if (!data.clickablePoints.some(cp => cp.maskIndex === foregroundPointIndex)) {
            data.clickablePoints.push({
              x: drawX,
              y: drawY,
              radius: clickRadius,
              maskIndex: foregroundPointIndex
            });
          }
        }
      });
    }
  }

  function handleCanvasClick(event) {
    const canvas = event.target;
    const canvasId = canvas.id;
    const data = canvasDataStore[canvasId];
    if (!data || !data.clickablePoints) return;

    const rect = canvas.getBoundingClientRect();
    const clickScaleX = canvas.width / rect.width;
    const clickScaleY = canvas.height / rect.height;
    const clickX = (event.clientX - rect.left) * clickScaleX;
    const clickY = (event.clientY - rect.top) * clickScaleY;

    for (let i = data.clickablePoints.length - 1; i >= 0; i--) {
      const pt = data.clickablePoints[i];

      const distance = Math.sqrt(
        Math.pow(clickX - pt.x, 2) + Math.pow(clickY - pt.y, 2),
      );

      // Check if click is within the point's clickable radius and it has a valid maskIndex
      if (distance <= pt.radius && pt.maskIndex !== undefined && pt.maskIndex >= 0) {
        clickedOnPoint = true;

        const targetMaskIndex = pt.maskIndex;

        if (data.masks && targetMaskIndex < data.masks.length) {
          const targetMask = data.masks[targetMaskIndex];
          const targetInstanceId = targetMask.instance_id;
          const checkboxId = `${canvasId}-mask-${targetInstanceId}`;
          const checkbox = document.getElementById(checkboxId);

          if (checkbox) {
            checkbox.checked = !checkbox.checked;
            redrawCanvas(canvasId); // Redraw to show/hide the mask
          } else {
            console.warn(`Could not find checkbox with ID: ${checkboxId} for mask index ${targetMaskIndex}`);
          }
        } else {
          console.warn(`Invalid mask index (${targetMaskIndex}) or mask list missing for canvas ${canvasId}`);
        }

        break;
      }
    }
  }
});
