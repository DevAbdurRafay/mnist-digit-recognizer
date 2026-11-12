(function () {
  'use strict';

  /* ── Canvas setup ──────────────────────────────────── */
  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  let brushSize = 18;
  let isDrawing = false;
  let hasDrawn = false;
  let points = [];

  function fillBlack() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, W, H);
  }
  fillBlack();

  function getScaledPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = W / rect.width;
    const scaleY = H / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  function startStroke(e) {
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    isDrawing = true;
    const pos = getScaledPos(e);
    points = [pos];
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = 'rgba(255,255,255,0.4)';
    ctx.shadowBlur = 6;
    ctx.fill();
    ctx.shadowBlur = 0;
    hasDrawn = true;
    updatePredictBtn();
  }

  function continueStroke(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const pos = getScaledPos(e);
    points.push(pos);

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(255,255,255,0.35)';
    ctx.shadowBlur = 7;

    if (points.length === 2) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      ctx.lineTo(points[1].x, points[1].y);
      ctx.stroke();
    } else if (points.length >= 3) {
      const len = points.length;
      const p0 = points[len - 3];
      const p1 = points[len - 2];
      const p2 = points[len - 1];
      const mx1 = (p0.x + p1.x) / 2;
      const my1 = (p0.y + p1.y) / 2;
      const mx2 = (p1.x + p2.x) / 2;
      const my2 = (p1.y + p2.y) / 2;
      ctx.beginPath();
      ctx.moveTo(mx1, my1);
      ctx.quadraticCurveTo(p1.x, p1.y, mx2, my2);
      ctx.stroke();
    }

    ctx.shadowBlur = 0;
  }

  function endStroke(e) {
    if (!isDrawing) return;
    e.preventDefault();
    isDrawing = false;
    points = [];
  }

  canvas.addEventListener('pointerdown', startStroke);
  canvas.addEventListener('pointermove', continueStroke);
  canvas.addEventListener('pointerup', endStroke);
  canvas.addEventListener('pointercancel', endStroke);
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  /* ── Prevent page scroll / zoom while drawing ───── */
  document.addEventListener('touchmove', (e) => {
    if (e.target === canvas) e.preventDefault();
  }, { passive: false });

  /* ── Brush pills ───────────────────────────────────── */
  document.querySelectorAll('.pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      brushSize = parseInt(pill.dataset.size, 10);
      document.querySelectorAll('.pill').forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');
    });
  });

  /* ── Clear ─────────────────────────────────────────── */
  const clearBtn = document.getElementById('clearBtn');
  const predictBtn = document.getElementById('predictBtn');

  function updatePredictBtn() {
    predictBtn.disabled = !hasDrawn;
  }

  clearBtn.addEventListener('click', () => {
    fillBlack();
    hasDrawn = false;
    updatePredictBtn();
    showState('ready');
  });

  /* ── Panel state machine ───────────────────────────── */
  const states = ['ready', 'analyzing', 'result', 'error'];
  const stateEls = {
    ready: document.getElementById('stateReady'),
    analyzing: document.getElementById('stateAnalyzing'),
    result: document.getElementById('stateResult'),
    error: document.getElementById('stateError'),
  };

  function showState(name) {
    states.forEach((s) => stateEls[s].classList.add('hidden'));
    stateEls[name].classList.remove('hidden');
  }

  /* ── Step animation ────────────────────────────────── */
  const STEP_NAMES = [
    'Reading Canvas Image',
    'Converting to Grayscale',
    'Removing Noise',
    'Cropping the Digit',
    'Matching Stroke Thickness',
    'Resizing to 20×20',
    'Placing on 28×28 Canvas',
    'Centering by Center of Mass',
    'Normalizing Pixels (0 to 1)',
    'Flattening to 1D (784 values)',
    'Running Neural Network',
    'Applying Softmax',
  ];
  const TOTAL_STEPS = STEP_NAMES.length;
  const STEP_DURATION = 250;

  const stepCounter = document.getElementById('stepCounter');
  const progressBar = document.getElementById('progressBar');
  const stepItems = document.querySelectorAll('.step-item');

  function resetSteps() {
    stepItems.forEach((el) => {
      el.classList.remove('active', 'done', 'failed');
    });
    stepCounter.textContent = 'Step 1 / 12';
    progressBar.style.width = '0%';
  }

  function runStepAnimation(onAllDone) {
    resetSteps();
    let currentStep = 0;
    let stopped = false;
    let timerId = null;

    function advance() {
      if (stopped) return;
      if (currentStep > 0) {
        stepItems[currentStep - 1].classList.remove('active');
        stepItems[currentStep - 1].classList.add('done');
      }
      if (currentStep >= TOTAL_STEPS) {
        progressBar.style.width = '100%';
        stepCounter.textContent = `Step ${TOTAL_STEPS} / ${TOTAL_STEPS}`;
        onAllDone();
        return;
      }
      stepItems[currentStep].classList.add('active');
      stepItems[currentStep].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      stepCounter.textContent = `Step ${currentStep + 1} / ${TOTAL_STEPS}`;
      progressBar.style.width = `${((currentStep + 1) / TOTAL_STEPS) * 100}%`;
      currentStep++;
      timerId = setTimeout(advance, STEP_DURATION);
    }

    advance();

    return {
      stop() { stopped = true; clearTimeout(timerId); },
      failAt(stepIndex) {
        stopped = true;
        clearTimeout(timerId);
        for (let i = 0; i < stepIndex; i++) {
          stepItems[i].classList.remove('active');
          stepItems[i].classList.add('done');
        }
        stepItems[stepIndex].classList.remove('active');
        stepItems[stepIndex].classList.add('failed');
        stepCounter.textContent = `Step ${stepIndex + 1} / ${TOTAL_STEPS}`;
        progressBar.style.width = `${((stepIndex + 1) / TOTAL_STEPS) * 100}%`;
      },
    };
  }

  /* ── Prediction ─────────────────────────────────────── */
  let apiResult = null;
  let animDone = false;
  let apiDone = false;
  let currentAnimator = null;

  function maybeShowResult() {
    if (animDone && apiDone) {
      if (apiResult && !apiResult.error) {
        renderResult(apiResult);
      }
    }
  }

  predictBtn.addEventListener('click', () => {
    if (!hasDrawn) return;
    apiResult = null;
    animDone = false;
    apiDone = false;

    showState('analyzing');
    predictBtn.disabled = true;

    const imageData = canvas.toDataURL('image/png');

    currentAnimator = runStepAnimation(() => {
      animDone = true;
      maybeShowResult();
    });

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData }),
    })
      .then((res) => {
        if (!res.ok) {
          return res.json().then((data) => {
            throw new Error((data.detail && data.detail.error) || data.error || `HTTP ${res.status}`);
          });
        }
        return res.json();
      })
      .then((data) => {
        apiResult = data;
        apiDone = true;
        maybeShowResult();
      })
      .catch((err) => {
        if (currentAnimator) currentAnimator.stop();
        showError(err.message || 'Network error');
      });
  });

  document.getElementById('retryBtn').addEventListener('click', () => {
    showState('ready');
    updatePredictBtn();
  });

  /* ── Render result ──────────────────────────────────── */
  function renderResult(data) {
    const { predicted_digit, confidence, probabilities } = data;

    document.getElementById('resultDigit').textContent = predicted_digit;
    document.getElementById('confidenceBadge').textContent = `${confidence.toFixed(1)}% Confidence`;

    const rows = document.querySelectorAll('.prob-row');
    rows.forEach((row) => {
      const digit = parseInt(row.dataset.digit, 10);
      const pct = probabilities[digit];
      const fill = row.querySelector('.prob-bar-fill');
      const pctEl = row.querySelector('.prob-pct');

      row.classList.toggle('winner', digit === predicted_digit);
      fill.style.width = '0%';
      pctEl.textContent = `${pct.toFixed(1)}%`;

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          fill.style.width = `${pct}%`;
        });
      });
    });

    showState('result');
    predictBtn.disabled = false;
  }

  function showError(msg) {
    document.getElementById('errorMsg').textContent = msg;
    showState('error');
    predictBtn.disabled = false;
  }
})();
