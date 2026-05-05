(() => {
    /* ─── Element References ─── */
    const video          = document.getElementById('webcam');
    const overlay        = document.getElementById('overlay');
    const placeholder    = document.getElementById('placeholder');
    const scanLine       = document.getElementById('scanLine');
    const startBtn       = document.getElementById('startBtn');
    const stopBtn        = document.getElementById('stopBtn');
    const resetBtn       = document.getElementById('resetBtn');
    const wordEl         = document.getElementById('currentWord');
    const sentenceEl     = document.getElementById('sentence');
    const confBar        = document.getElementById('confBar');
    const confVal        = document.getElementById('confVal');
    const bufferSizeEl   = document.getElementById('bufferSize');
    const bufferSegs     = document.getElementById('bufferSegments');
    const historyEl      = document.getElementById('history');
    const historyCount   = document.getElementById('historyCount');
    const latencyBadge   = document.getElementById('latencyBadge');
    const autoplay       = document.getElementById('autoplay');
    const player         = document.getElementById('player');
    const statusPill     = document.getElementById('statusPill');
    const statusText     = document.getElementById('statusText');
    const copyBtn        = document.getElementById('copySentenceBtn');
    const toast          = document.getElementById('toast');

    /* Pipeline stages */
    const stages = {
        yolo:  document.getElementById('stage-yolo'),
        mp:    document.getElementById('stage-mp'),
        lstm:  document.getElementById('stage-lstm'),
        nlp:   document.getElementById('stage-nlp'),
        tts:   document.getElementById('stage-tts'),
    };

    /* ─── Config ─── */
    const FRAME_RATE_MS  = 200;    // 5 fps
    const JPEG_QUALITY   = 0.72;
    const SCALED_WIDTH   = 480;

    const HAND_CONNECTIONS = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[17,18],[18,19],[19,20],
        [0,17],
    ];

    /* ─── Session ID ─── */
    const sessionId = (() => {
        let sid = localStorage.getItem('sibindo_session');
        if (!sid) {
            sid = 'sess_' + Math.random().toString(36).slice(2, 12);
            localStorage.setItem('sibindo_session', sid);
        }
        return sid;
    })();

    /* ─── State ─── */
    let stream        = null;
    let timer         = null;
    let inflight      = false;
    let lastAudioUrl  = null;
    let lastFrameSize = { w: 0, h: 0 };
    let lastWord      = null;
    let toastTimer    = null;

    /* ─── Camera ─── */
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false,
            });
            video.srcObject = stream;
            await video.play();

            placeholder.style.display    = 'none';
            scanLine.classList.remove('hidden');
            startBtn.disabled            = true;
            stopBtn.disabled             = false;

            setStatus('active', 'Aktif');
            timer = setInterval(captureAndSend, FRAME_RATE_MS);
        } catch (err) {
            setStatus('error', 'Error');
            showToast('Gagal kamera: ' + err.message, 3000);
        }
    }

    function stopCamera() {
        if (timer)  { clearInterval(timer); timer = null; }
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }

        video.srcObject = null;
        placeholder.style.display = 'grid';
        scanLine.classList.add('hidden');
        startBtn.disabled = false;
        stopBtn.disabled  = true;

        setStatus('idle', 'Siap');
        clearOverlay();
        setAllStages(false);
    }

    /* ─── Capture & Predict ─── */
    async function captureAndSend() {
        if (inflight || !video.videoWidth) return;
        inflight = true;

        const cvs   = document.createElement('canvas');
        const scale = SCALED_WIDTH / video.videoWidth;
        cvs.width   = SCALED_WIDTH;
        cvs.height  = Math.round(video.videoHeight * scale);
        cvs.getContext('2d').drawImage(video, 0, 0, cvs.width, cvs.height);
        const dataUrl = cvs.toDataURL('image/jpeg', JPEG_QUALITY);
        lastFrameSize = { w: cvs.width, h: cvs.height };

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame: dataUrl, session_id: sessionId }),
            });
            if (res.ok) renderResult(await res.json());
        } catch (e) {
            console.warn('[sibindo] predict failed', e);
        } finally {
            inflight = false;
        }
    }

    /* ─── Render Result ─── */
    function renderResult(data) {
        const buf  = data.buffer_size ?? 0;
        const conf = Math.round((data.confidence ?? 0) * 100);
        const ms   = data.latency_ms;

        /* Latency badge */
        latencyBadge.textContent = ms ? `${ms} ms` : '— ms';
        latencyBadge.classList.toggle('fast', ms !== undefined && ms < 100);
        latencyBadge.classList.toggle('slow', ms !== undefined && ms >= 200);

        /* Buffer */
        bufferSizeEl.textContent = buf;
        updateBufferSegs(buf);

        /* Confidence bar */
        confBar.style.width     = `${conf}%`;
        confVal.textContent     = `${conf}%`;

        /* Word */
        const word = data.word;
        if (word && word !== lastWord) {
            lastWord = word;
            wordEl.textContent = word;
            wordEl.classList.remove('muted', 'word-new');
            void wordEl.offsetWidth;   // reflow to restart animation
            wordEl.classList.add('word-new');
        } else if (!word && data.detected) {
            if (lastWord !== '__scanning__') {
                lastWord = '__scanning__';
                wordEl.textContent = 'Mendeteksi...';
                wordEl.classList.add('muted');
                wordEl.classList.remove('word-new');
            }
        } else if (!data.detected) {
            if (lastWord !== null) {
                lastWord = null;
                wordEl.textContent = 'Menunggu input...';
                wordEl.classList.add('muted');
                wordEl.classList.remove('word-new');
            }
        }

        /* Sentence */
        const sent = data.sentence || '';
        sentenceEl.textContent = sent || '—';
        sentenceEl.classList.toggle('muted', !sent);
        if (copyBtn) copyBtn.disabled = !sent;

        /* History */
        renderHistory(data.history || []);

        /* Audio */
        if (data.audio_url && data.audio_url !== lastAudioUrl) {
            lastAudioUrl = data.audio_url;
            player.src   = data.audio_url;
            if (autoplay.checked) player.play().catch(() => {});
        }

        /* Pipeline stages */
        const hasHand  = data.landmarks?.hands?.length > 0;
        const hasWord  = !!data.word;
        const hasSent  = !!sent;
        const hasAudio = !!data.audio_url;

        setStage('yolo',  !!data.detected || !!data.bbox);
        setStage('mp',    hasHand);
        setStage('lstm',  hasWord);
        setStage('nlp',   hasSent);
        setStage('tts',   hasAudio && data.audio_url === lastAudioUrl && player.currentTime > 0);

        /* Overlay */
        drawOverlay(data);
    }

    /* ─── Segment Buffer ─── */
    function updateBufferSegs(n) {
        if (!bufferSegs) return;
        const segs = bufferSegs.querySelectorAll('.seg');
        segs.forEach((seg, i) => {
            seg.classList.toggle('filled',  i < n);
            seg.classList.toggle('active',  i === n - 1 && n > 0);
        });
    }

    /* ─── History Chips ─── */
    function renderHistory(words) {
        if (!historyEl) return;
        if (!words.length) {
            historyEl.innerHTML = '<li class="chip-empty">Belum ada kata terdeteksi</li>';
            if (historyCount) historyCount.textContent = '0';
            return;
        }
        if (historyCount) historyCount.textContent = words.length;
        historyEl.innerHTML = words
            .slice().reverse()
            .map((w, i) => `<li class="chip" style="animation-delay:${i * 25}ms">${escapeHtml(w)}</li>`)
            .join('');
    }

    /* ─── Pipeline Stages ─── */
    function setStage(key, lit) {
        stages[key]?.classList.toggle('lit', lit);
    }

    function setAllStages(lit) {
        Object.keys(stages).forEach(k => setStage(k, lit));
    }

    /* ─── Status Pill ─── */
    function setStatus(state, label) {
        statusPill.className = `status-pill ${state}`;
        if (statusText) statusText.textContent = label;
    }

    /* ─── Overlay Drawing ─── */
    function clearOverlay() {
        const ctx  = overlay.getContext('2d');
        overlay.width  = video.clientWidth;
        overlay.height = video.clientHeight;
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    function drawOverlay(data) {
        const ctx = overlay.getContext('2d');
        overlay.width  = video.clientWidth;
        overlay.height = video.clientHeight;
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        const ow = overlay.width;
        const oh = overlay.height;

        /* Bounding box */
        if (data.bbox) {
            const sx = ow / lastFrameSize.w;
            const sy = oh / lastFrameSize.h;
            const [x1, y1, x2, y2] = data.bbox;
            const bx = x1 * sx, by = y1 * sy;
            const bw = (x2 - x1) * sx, bh = (y2 - y1) * sy;

            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth   = 2;
            ctx.shadowColor = '#22c55e';
            ctx.shadowBlur  = 8;

            /* Corner brackets instead of full rect */
            const cs = Math.min(bw, bh) * 0.14;
            const corners = [
                [bx, by, cs, 0, 0, cs],
                [bx + bw, by, -cs, 0, 0, cs],
                [bx, by + bh, cs, 0, 0, -cs],
                [bx + bw, by + bh, -cs, 0, 0, -cs],
            ];
            corners.forEach(([cx, cy, dx1, dy1, dx2, dy2]) => {
                ctx.beginPath();
                ctx.moveTo(cx + dx1, cy + dy1);
                ctx.lineTo(cx, cy);
                ctx.lineTo(cx + dx2, cy + dy2);
                ctx.stroke();
            });
            ctx.shadowBlur = 0;
        }

        /* Hand landmarks */
        if (data.landmarks?.hands) {
            const handColors = ['#00e5b0', '#60a5fa'];
            data.landmarks.hands.forEach((hand, hi) => {
                const pts   = hand.points;
                const color = handColors[hi % 2];

                /* Connections */
                ctx.strokeStyle = color;
                ctx.lineWidth   = 1.5;
                ctx.globalAlpha = 0.75;
                HAND_CONNECTIONS.forEach(([a, b]) => {
                    if (!pts[a] || !pts[b]) return;
                    ctx.beginPath();
                    ctx.moveTo(pts[a][0] * ow, pts[a][1] * oh);
                    ctx.lineTo(pts[b][0] * ow, pts[b][1] * oh);
                    ctx.stroke();
                });

                /* Knuckle dots */
                ctx.globalAlpha = 1;
                pts.forEach((p, pi) => {
                    const isKey = [0, 4, 8, 12, 16, 20].includes(pi);
                    ctx.beginPath();
                    ctx.arc(p[0] * ow, p[1] * oh, isKey ? 4 : 2.5, 0, Math.PI * 2);
                    ctx.fillStyle = isKey ? '#ffffff' : color;
                    ctx.shadowColor = color;
                    ctx.shadowBlur  = isKey ? 6 : 3;
                    ctx.fill();
                    ctx.shadowBlur = 0;
                });
            });
            ctx.globalAlpha = 1;
        }
    }

    /* ─── Reset ─── */
    async function resetAll() {
        await fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        });
        lastWord     = null;
        lastAudioUrl = null;
        wordEl.textContent = 'Menunggu input...';
        wordEl.classList.add('muted');
        wordEl.classList.remove('word-new');
        sentenceEl.textContent = '—';
        sentenceEl.classList.add('muted');
        confBar.style.width = '0%';
        if (confVal) confVal.textContent = '0%';
        bufferSizeEl.textContent = '0';
        updateBufferSegs(0);
        renderHistory([]);
        if (copyBtn) copyBtn.disabled = true;
        player.removeAttribute('src');
        setAllStages(false);
        showToast('Reset berhasil');
    }

    /* ─── Copy Sentence ─── */
    if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
            const txt = sentenceEl.textContent;
            if (!txt || txt === '—') return;
            try {
                await navigator.clipboard.writeText(txt);
                copyBtn.classList.add('copied');
                showToast('Disalin!');
                setTimeout(() => copyBtn.classList.remove('copied'), 1500);
            } catch {
                showToast('Gagal menyalin');
            }
        });
    }

    /* ─── Toast ─── */
    function showToast(msg, duration = 1800) {
        if (toastTimer) { clearTimeout(toastTimer); toast.className = 'toast'; }
        toast.textContent = msg;
        toast.classList.add('show');
        toastTimer = setTimeout(() => {
            toast.classList.remove('show');
            toast.classList.add('hide');
            setTimeout(() => { toast.className = 'toast'; }, 350);
        }, duration);
    }

    /* ─── Utility ─── */
    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
    }

    /* ─── Event Listeners ─── */
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click',  stopCamera);
    resetBtn.addEventListener('click', resetAll);
})();
