document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsSection = document.getElementById('results-section');
    const loader = document.querySelector('.loader');
    const btnText = document.querySelector('.btn-text');

    let currentFile = null;

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');

        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            analyzeBtn.classList.remove('hidden');
            resultsSection.classList.add('hidden'); // query hidden if re-uploading
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering dropZone click if nested
        currentFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        resultsSection.classList.add('hidden');
    });

    // Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        setLoading(true);

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Analysis failed');

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error(Error, error);
            alert('An error occurred during analysis.');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.classList.add('disabled');
            analyzeBtn.disabled = true;
            loader.classList.remove('hidden');
            btnText.textContent = 'Analyzing...';
        } else {
            analyzeBtn.classList.remove('disabled');
            analyzeBtn.disabled = false;
            loader.classList.add('hidden');
            btnText.textContent = 'Analyze Image';
        }
    }

    function displayResults(data) {
        resultsSection.classList.remove('hidden');

        // Update Text
        const predText = document.getElementById('prediction-text');
        predText.textContent = data.prediction;

        // Confidence Ring
        const circle = document.getElementById('confidence-circle');
        const confText = document.getElementById('confidence-text');
        const percentage = data.confidence;

        // circumference of circle with r=15.9155 is ~100
        const strokeDash = `${percentage}, 100`;

        // Reset animation
        circle.style.strokeDasharray = "0, 100";
        setTimeout(() => {
            circle.style.strokeDasharray = strokeDash;
        }, 100);

        confText.textContent = `${Math.round(percentage)}%`;

        // Metrics
        document.getElementById('metric-sharpness').textContent = data.metrics.sharpness;

        // Show Explanation (Replace Preview)
        const imagePreview = document.getElementById('image-preview');

        if (data.explanation_url) {
            imagePreview.src = data.explanation_url;
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
});
