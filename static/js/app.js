/* ----------------------------------------------------
   MED-X AI - Frontend Application Controller
---------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Selectors
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const sampleBtns = document.querySelectorAll(".sample-btn");
    const opacitySlider = document.getElementById("opacity-slider");
    const opacityVal = document.getElementById("opacity-val");
    const viewModeBtns = document.querySelectorAll(".view-mode-btn");
    
    const visualizerWelcome = document.getElementById("visualizer-welcome");
    const loadingOverlay = document.getElementById("loading-overlay");
    const imageWrapper = document.getElementById("image-wrapper");
    const scanline = document.getElementById("scanline");
    
    // View containers
    const viewOverlay = document.getElementById("view-overlay");
    const viewSplit = document.getElementById("view-split");
    const viewSide = document.getElementById("view-side");
    
    // Images elements
    const imgOrigOverlay = document.getElementById("img-orig-overlay");
    const imgCamOverlay = document.getElementById("img-cam-overlay");
    const imgOrigSplit = document.getElementById("img-orig-split");
    const imgCamSplit = document.getElementById("img-cam-split");
    const imgOrigSide = document.getElementById("img-orig-side");
    const imgBgSide = document.getElementById("img-bg-side");
    const imgCamSide = document.getElementById("img-cam-side");
    
    // Split Screen
    const splitDivider = document.getElementById("split-divider");
    const splitRight = viewSplit.querySelector(".split-right"); // Contains the CAM
    
    // Pixel Inspector
    const pixelInspector = document.getElementById("pixel-inspector");
    const inspectPos = document.getElementById("inspect-pos");
    const inspectVal = document.getElementById("inspect-val");
    
    // Diagnostics Panel
    const diagnosticsPlaceholder = document.getElementById("diagnostics-placeholder");
    const diagnosticsContent = document.getElementById("diagnostics-content");
    const diagName = document.getElementById("diag-name");
    const diagSeverity = document.getElementById("diag-severity");
    const diagConfidencePct = document.getElementById("diag-confidence-pct");
    const diagConfidenceFill = document.getElementById("diag-confidence-fill");
    const breakdownList = document.getElementById("breakdown-list");
    const diagExplanation = document.getElementById("diag-explanation");
    const reportRef = document.getElementById("report-ref");
    const reportDate = document.getElementById("report-date");
    const reportFindings = document.getElementById("report-findings");
    const reportRecommendations = document.getElementById("report-recommendations");
    const reportPrecautions = document.getElementById("report-precautions");
    
    // Actions Buttons
    const btnCopyReport = document.getElementById("btn-copy-report");
    const btnDownloadPdf = document.getElementById("btn-download-pdf");
    const visualizerFooter = document.getElementById("visualizer-footer");
    
    // Global State variables
    let currentMode = "overlay"; // overlay, split, side
    let isDraggingSplit = false;
    let globalRawGrid = null;
    let activeAnalysisData = null;
    
    // Generate random reference ID for patient report
    function generateRefId() {
        const chars = "0123456789ABCDEF";
        let ref = "#";
        for (let i = 0; i < 6; i++) {
            ref += chars[Math.floor(Math.random() * 16)];
        }
        return ref;
    }
    
    // Format current date
    function getFormattedDate() {
        const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
        return new Date().toLocaleDateString("en-US", options);
    }
    
    /* ----------------------------------------------------
       Drag and Drop Upload Handlers
    ---------------------------------------------------- */
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });
    
    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });
    
    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
    
    uploadZone.addEventListener("click", () => {
        fileInput.click();
    });
    
    function handleFileUpload(file) {
        // Deselect any selected sample button
        sampleBtns.forEach(btn => btn.classList.remove("selected"));
        
        // Form body data
        const formData = new FormData();
        formData.append("file", file);
        
        analyzeRadiograph(formData, true);
    }
    
    /* ----------------------------------------------------
       Quick-Select Samples
    ---------------------------------------------------- */
    sampleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            // Manage UI states
            sampleBtns.forEach(b => b.classList.remove("selected"));
            btn.classList.add("selected");
            
            const sampleName = btn.getAttribute("data-sample");
            
            // JSON body data
            const bodyData = JSON.stringify({ sample_name: sampleName });
            analyzeRadiograph(bodyData, false);
        });
    });
    
    /* ----------------------------------------------------
       AI Analysis Routine
    ---------------------------------------------------- */
    function analyzeRadiograph(bodyData, isFile) {
        // Show loading screen, hide results
        visualizerWelcome.style.display = "none";
        loadingOverlay.style.display = "flex";
        imageWrapper.style.display = "none";
        visualizerFooter.style.display = "none";
        diagnosticsPlaceholder.style.display = "flex";
        diagnosticsContent.style.display = "none";
        pixelInspector.style.display = "none";
        
        // Progress bar simulation
        const progressBar = document.getElementById("analysis-progress-bar");
        progressBar.style.width = "0%";
        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.floor(Math.random() * 8) + 2;
                if (progress > 90) progress = 90;
                progressBar.style.width = progress + "%";
            }
        }, 150);
        
        const headers = isFile ? {} : { "Content-Type": "application/json" };
        
        fetch("/api/analyze", {
            method: "POST",
            headers: headers,
            body: bodyData
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(progressInterval);
            progressBar.style.width = "100%";
            
            setTimeout(() => {
                if (data.success) {
                    displayAnalysisResults(data);
                } else {
                    alert("Analysis Failed: " + data.error);
                    resetUI();
                }
            }, 300);
        })
        .catch(error => {
            clearInterval(progressInterval);
            console.error("Error during analysis:", error);
            alert("An error occurred during server communication.");
            resetUI();
        });
    }
    
    function resetUI() {
        loadingOverlay.style.display = "none";
        visualizerWelcome.style.display = "flex";
        imageWrapper.style.display = "none";
        visualizerFooter.style.display = "none";
        diagnosticsPlaceholder.style.display = "flex";
        diagnosticsContent.style.display = "none";
        sampleBtns.forEach(btn => btn.classList.remove("selected"));
    }
    
    /* ----------------------------------------------------
       Display Results & Inject DOM Elements
    ---------------------------------------------------- */
    function displayAnalysisResults(data) {
        activeAnalysisData = data;
        globalRawGrid = data.raw_grid;
        
        // Hide loader overlay, show elements
        loadingOverlay.style.display = "none";
        imageWrapper.style.display = "block";
        visualizerFooter.style.display = "flex";
        diagnosticsPlaceholder.style.display = "none";
        diagnosticsContent.style.display = "flex";
        
        // 1. Inject Images into all view modes
        // Overlay Mode
        imgOrigOverlay.src = data.original_image;
        imgCamOverlay.src = data.heatmap_image;
        
        // Split Mode
        imgOrigSplit.src = data.original_image;
        imgCamSplit.src = data.heatmap_image;
        
        // Side Mode
        imgOrigSide.src = data.original_image;
        imgBgSide.src = data.original_image;
        imgCamSide.src = data.heatmap_image;
        
        // Trigger scan sweep animation
        scanline.classList.add("active");
        setTimeout(() => {
            scanline.classList.remove("active");
        }, 2200);
        
        // Force images alignment sizes inside Split Mode
        imgCamSplit.style.width = imageWrapper.clientWidth + "px";
        imgCamSplit.style.height = imageWrapper.clientHeight + "px";
        
        // 2. Adjust Opacity blend
        applyOpacityBlend();
        
        // 3. Reset Split Divider to center
        resetSplitDivider();
        
        // 4. Inject Diagnostics Information
        diagName.textContent = data.diagnosis;
        
        // Set severity badge
        const severity = data.clinical_report.severity;
        diagSeverity.textContent = severity;
        diagSeverity.className = "severity-badge"; // Reset classes
        if (severity === "Normal") {
            diagSeverity.classList.add("badge-normal");
        } else if (severity.includes("Moderate") && !severity.includes("High")) {
            diagSeverity.classList.add("badge-warning");
        } else {
            diagSeverity.classList.add("badge-danger");
        }
        
        // Confidence percentage
        const confPct = (data.confidence * 100).toFixed(1);
        diagConfidencePct.textContent = confPct + "%";
        diagConfidenceFill.style.width = confPct + "%";
        
        // Class probability breakdown list
        breakdownList.innerHTML = "";
        const sortedBreakdown = Object.entries(data.breakdown)
            .sort((a, b) => b[1] - a[1]); // Sort highest first
            
        sortedBreakdown.forEach(([clsName, prob]) => {
            const probPct = (prob * 100).toFixed(1);
            const isHigh = clsName === data.diagnosis;
            
            const breakdownItem = document.createElement("div");
            breakdownItem.className = `breakdown-item ${isHigh ? "high-prob" : ""}`;
            breakdownItem.innerHTML = `
                <div class="breakdown-info">
                    <span class="breakdown-label">${clsName}</span>
                    <span class="breakdown-pct">${probPct}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: ${probPct}%"></div>
                </div>
            `;
            breakdownList.appendChild(breakdownItem);
        });
        
        // Explainability Text
        diagExplanation.textContent = data.clinical_report.explanation;
        
        // Structured Patient Radiograph Report
        reportRef.textContent = generateRefId();
        reportDate.textContent = getFormattedDate();
        reportFindings.textContent = data.clinical_report.findings;
        
        // Recommendations list
        reportRecommendations.innerHTML = "";
        data.clinical_report.recommendations.forEach(rec => {
            const li = document.createElement("li");
            li.textContent = rec;
            reportRecommendations.appendChild(li);
        });
        
        // Patient precautions list
        reportPrecautions.innerHTML = "";
        data.clinical_report.precautions.forEach(prec => {
            const li = document.createElement("li");
            li.textContent = prec;
            reportPrecautions.appendChild(li);
        });
    }
    
    /* ----------------------------------------------------
       Opacity Blending Slider Logic
    ---------------------------------------------------- */
    opacitySlider.addEventListener("input", () => {
        applyOpacityBlend();
    });
    
    function applyOpacityBlend() {
        const val = opacitySlider.value;
        opacityVal.textContent = val + "%";
        const opacityDec = val / 100;
        
        if (imgCamOverlay) {
            imgCamOverlay.style.opacity = opacityDec;
        }
        if (imgCamSide) {
            imgCamSide.style.opacity = opacityDec;
        }
    }
    
    /* ----------------------------------------------------
       View Mode Selector Toggle
    ---------------------------------------------------- */
    viewModeBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            viewModeBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const mode = btn.getAttribute("data-mode");
            currentMode = mode;
            
            // Hide all views
            viewOverlay.style.display = "none";
            viewSplit.style.display = "none";
            viewSide.style.display = "none";
            
            // Show target view
            if (mode === "overlay") {
                viewOverlay.style.display = "block";
            } else if (mode === "split") {
                viewSplit.style.display = "block";
                resetSplitDivider();
            } else if (mode === "side") {
                viewSide.style.display = "flex";
            }
        });
    });
    
    /* ----------------------------------------------------
       Split Screen Slider Interaction
    ---------------------------------------------------- */
    function resetSplitDivider() {
        if (currentMode !== "split") return;
        splitDivider.style.left = "50%";
        splitRight.style.width = "50%"; // Left width crop
        
        // Adjust the cropped image inside to retain full visual alignment
        const rect = imageWrapper.getBoundingClientRect();
        imgCamSplit.style.width = rect.width + "px";
        imgCamSplit.style.height = rect.height + "px";
    }
    
    splitDivider.addEventListener("mousedown", (e) => {
        isDraggingSplit = true;
        e.preventDefault();
    });
    
    // Touch screen support
    splitDivider.addEventListener("touchstart", (e) => {
        isDraggingSplit = true;
    });
    
    window.addEventListener("mouseup", () => {
        isDraggingSplit = false;
    });
    
    window.addEventListener("touchend", () => {
        isDraggingSplit = false;
    });
    
    window.addEventListener("mousemove", (e) => {
        if (!isDraggingSplit) return;
        handleSplitDrag(e.clientX);
    });
    
    window.addEventListener("touchmove", (e) => {
        if (!isDraggingSplit) return;
        if (e.touches.length > 0) {
            handleSplitDrag(e.touches[0].clientX);
        }
    });
    
    function handleSplitDrag(clientX) {
        const rect = imageWrapper.getBoundingClientRect();
        let relativeX = clientX - rect.left;
        
        // Clamp boundaries
        relativeX = Math.max(0, Math.min(relativeX, rect.width));
        
        const percentage = (relativeX / rect.width) * 100;
        
        splitDivider.style.left = percentage + "%";
        // The splitRight container holds the overlay and crops it. 
        // If we set splitRight width to percentage%, it covers the left side.
        splitRight.style.width = percentage + "%";
        
        // Ensure image inside has exact dimensions of the outer bounding box
        imgCamSplit.style.width = rect.width + "px";
        imgCamSplit.style.height = rect.height + "px";
    }
    
    // Handle window resize (keep images synchronized)
    window.addEventListener("resize", () => {
        if (activeAnalysisData && currentMode === "split") {
            resetSplitDivider();
        }
    });
    
    /* ----------------------------------------------------
       Pixel Inspector Value Hover Logic
    ---------------------------------------------------- */
    imageWrapper.addEventListener("mousemove", (e) => {
        if (!activeAnalysisData || !globalRawGrid) return;
        
        // Split mode divider drag shouldn't trigger inspector popups
        if (isDraggingSplit) return;
        
        const rect = imageWrapper.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        // Verify boundaries
        if (mouseX < 0 || mouseX > rect.width || mouseY < 0 || mouseY > rect.height) {
            pixelInspector.style.display = "none";
            return;
        }
        
        // Map to 7x7 grid coordinates
        const col = Math.min(6, Math.floor((mouseX / rect.width) * 7));
        const row = Math.min(6, Math.floor((mouseY / rect.height) * 7));
        
        const activationVal = globalRawGrid[row][col];
        
        // Show tooltip
        pixelInspector.style.display = "flex";
        pixelInspector.style.left = (mouseX + 15) + "px";
        pixelInspector.style.top = (mouseY + 15) + "px";
        
        inspectPos.textContent = `X: ${col}, Y: ${row}`;
        inspectVal.textContent = activationVal.toFixed(3);
    });
    
    imageWrapper.addEventListener("mouseleave", () => {
        pixelInspector.style.display = "none";
    });
    
    /* ----------------------------------------------------
       Report Actions: Copy and Download Text
    ---------------------------------------------------- */
    btnCopyReport.addEventListener("click", () => {
        if (!activeAnalysisData) return;
        
        const rText = generateReportTextFormat();
        navigator.clipboard.writeText(rText)
            .then(() => {
                const origHtml = btnCopyReport.innerHTML;
                btnCopyReport.innerHTML = `<i class="fa-solid fa-check"></i> Report Copied!`;
                btnCopyReport.style.borderColor = "var(--color-teal)";
                btnCopyReport.style.color = "var(--color-teal)";
                
                setTimeout(() => {
                    btnCopyReport.innerHTML = origHtml;
                    btnCopyReport.style.borderColor = "";
                    btnCopyReport.style.color = "";
                }, 2000);
            })
            .catch(err => {
                console.error("Failed to copy text:", err);
            });
    });
    
    btnDownloadPdf.addEventListener("click", () => {
        if (!activeAnalysisData) return;
        
        const rText = generateReportTextFormat();
        const blob = new Blob([rText], { type: "text/plain;charset=utf-8" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        
        const cleanDiag = activeAnalysisData.diagnosis.replace(/\s+/g, "_").toLowerCase();
        link.download = `MEDX_AI_Report_${reportRef.textContent.replace("#", "")}_${cleanDiag}.txt`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
    
    function generateReportTextFormat() {
        const refId = reportRef.textContent;
        const date = reportDate.textContent;
        const data = activeAnalysisData;
        const rep = data.clinical_report;
        
        let recText = "";
        rep.recommendations.forEach((rec, idx) => {
            recText += `${idx + 1}. ${rec}\n`;
        });
        
        let precText = "";
        rep.precautions.forEach((prec, idx) => {
            precText += `${idx + 1}. ${prec}\n`;
        });
        
        return `======================================================================
                     MED-X AI CLINICAL RADIOLOGY REPORT
======================================================================
Report Reference ID: ${refId}
Timestamp: ${date}
AI Processing Pipeline: DenseNet/ResNet-18 Feature Extractor
GRAD-CAM Saliency Engine: Active
----------------------------------------------------------------------
PRIMARY FINDING:      ${data.diagnosis.toUpperCase()}
SEVERITY LEVEL:       ${rep.severity.toUpperCase()}
DIAGNOSTIC CONFIDENCE: ${(data.confidence * 100).toFixed(2)}%
----------------------------------------------------------------------
FINDINGS & ANATOMICAL OBSERVATIONS:
${rep.findings}

PATHOPHYSIOLOGICAL INTERPRETATION:
${rep.explanation}

CLINICAL RECOMMENDATIONS & ACTION GUIDELINES:
${recText}

PATIENT SELF-CARE PRECAUTIONS & ADVICE:
${precText}
----------------------------------------------------------------------
CONFIDENCE DISTRIBUTION:
${Object.entries(data.breakdown)
    .sort((a,b) => b[1] - a[1])
    .map(([k,v]) => `- ${k.padEnd(20)}: ${(v*100).toFixed(2)}%`)
    .join('\n')}
----------------------------------------------------------------------
DISCLAIMER NOTICE:
This report is generated by a demonstration machine learning system.
Findings are for educational, research and auxiliary screening purposes
only. This report does not constitute official medical advice or diagnostic
determinations. Please correlate clinically with a board-certified radiologist.
======================================================================
`;
    }
});
