"""
VIDA + PANDA 웹 UI — 표준 라이브러리만 사용 (추가 설치 불필요).

    python vp_app.py

브라우저가 자동으로 http://localhost:8765 를 엽니다.
"""
from __future__ import annotations
import json, os, sys, threading, webbrowser, mimetypes, urllib.parse, socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vp_core as C

# .env 를 직접 파싱 (python-dotenv 없이)
ENV_DIAG = {'path': None, 'exists': False, 'encoding': None,
            'loaded_keys': [], 'error': None}

def load_env(path='.env'):
    p = Path(path).resolve()
    ENV_DIAG['path'] = str(p)
    if not p.exists():
        return
    ENV_DIAG['exists'] = True

    # 메모장이 UTF-16 / BOM 으로 저장하는 경우가 많아 인코딩을 차례로 시도
    text = None
    for enc in ('utf-8-sig', 'utf-16', 'cp949', 'utf-8'):
        try:
            text = p.read_text(encoding=enc)
            if '\x00' in text:      # UTF-16 을 잘못 읽은 경우
                continue
            ENV_DIAG['encoding'] = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        ENV_DIAG['error'] = '.env 를 읽을 수 없습니다 (인코딩 문제)'
        return

    for line in text.splitlines():
        line = line.strip().lstrip('\ufeff')
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip().lstrip('\ufeff')
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        os.environ[k] = v              # setdefault 대신 항상 덮어쓰기
        if v:
            ENV_DIAG['loaded_keys'].append(k)

load_env()

STATE = {'runner': None, 'thread': None}
PORT = int(os.getenv('VP_PORT', '8765'))


def providers_info():
    try:
        p = C.Providers()
        agents = p.list_agents()
        return {'agents': [{'name': k, 'provider': v} for k, v in agents.items()],
                'mock_agents': list(C.MockProviders.SKILL),
                'ollama_models': p.ollama_models,
                'notes': p.errors[:6],
                'env': ENV_DIAG}
    except Exception as ex:
        return {'agents': [], 'mock_agents': list(C.MockProviders.SKILL),
                'ollama_models': [], 'notes': [f'제공사 초기화 실패: {ex}'],
                'env': ENV_DIAG}


def snapshot():
    r = STATE['runner']
    if r is None:
        return {'phase': 'idle', 'done': 0, 'total': 0, 'log': [],
                'metrics': [], 'summary': {}, 'n_transcripts': 0}
    s = r.state
    return {'phase': s['phase'], 'done': s['done'], 'total': s['total'],
            'log': s['log'][-120:], 'metrics': s['metrics'],
            'summary': s.get('summary', {}), 'error': s.get('error'),
            'weights': s.get('weights', {}),
            'n_transcripts': len(s.get('transcripts', {}))}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a): pass          # 콘솔 소음 제거

    # ── 응답 헬퍼 ────────────────────────────────────────────────────────
    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, default=str).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    # ── GET ──────────────────────────────────────────────────────────────
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)

        if u.path == '/':
            return self._send(200, HTML, 'text/html; charset=utf-8')
        if u.path == '/api/providers':
            return self._send(200, providers_info())
        if u.path == '/api/state':
            return self._send(200, snapshot())
        if u.path == '/api/defaults':
            return self._send(200, {**C.DEFAULT_CFG,
                                    'dataset_root': os.getenv('CDDM_ROOT', './data/images'),
                                    'output_dir': os.getenv('VP_OUTPUT', './vida_panda_results')})

        if u.path == '/api/transcripts':
            r = STATE['runner']
            if not r: return self._send(200, [])
            out = []
            for name, t in r.state.get('transcripts', {}).items():
                out.append({'name': name, 'label': t.get('file_name', name), 'gt': t['gt'],
                            'consensus': t['consensus'], 'baseline': t['baseline'],
                            'ok': t['consensus'] == t['gt']})
            return self._send(200, out)

        if u.path == '/api/transcript':
            r = STATE['runner']; name = (q.get('name') or [''])[0]
            t = (r.state.get('transcripts', {}) if r else {}).get(name)
            return self._send(200, t or {})

        if u.path == '/img':
            r = STATE['runner']; name = (q.get('name') or [''])[0]
            t = (r.state.get('transcripts', {}) if r else {}).get(name)
            if not t: return self._send(404, {'error': 'not found'})
            p = Path(t['image_path'])
            if not p.exists(): return self._send(404, {'error': 'missing file'})
            ctype = mimetypes.guess_type(str(p))[0] or 'application/octet-stream'
            return self._send(200, p.read_bytes(), ctype)

        return self._send(404, {'error': 'not found'})

    # ── POST ─────────────────────────────────────────────────────────────
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(n) or b'{}')

        if u.path == '/api/start':
            r = STATE['runner']
            if r and r.state['phase'] not in ('idle', 'done', 'error', 'stopped'):
                return self._send(409, {'error': '이미 실행 중입니다.'})
            cfg = {**C.DEFAULT_CFG, **body}
            for k in ('n_images', 'n_debate', 'seed'): cfg[k] = int(cfg[k])
            cfg['softmax_temp'] = float(cfg['softmax_temp'])
            runner = C.Runner(cfg)
            STATE['runner'] = runner
            th = threading.Thread(target=runner.run, daemon=True)
            STATE['thread'] = th; th.start()
            return self._send(200, {'ok': True})

        if u.path == '/api/stop':
            r = STATE['runner']
            if r: r.stop_flag.set()
            return self._send(200, {'ok': True})

        return self._send(404, {'error': 'not found'})


HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VIDA + PANDA</title>
<style>
:root{
  --bg:#f6f7f9; --panel:#fff; --ink:#16181d; --muted:#6b7280; --line:#e3e6ea;
  --accent:#2a6f97; --good:#2a9d8f; --bad:#e06c5a; --warn:#e9a23b;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#14161a; --panel:#1c1f25; --ink:#e8eaed; --muted:#98a0ab; --line:#2c313a;
  --accent:#7cb7d8;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.6 system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}
header{padding:18px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0;font-size:17px;letter-spacing:-.01em}
h1 small{color:var(--muted);font-weight:400;font-size:13px;margin-left:8px}
.wrap{display:grid;grid-template-columns:320px 1fr;gap:16px;padding:16px;
  max-width:1500px;margin:0 auto}
@media (max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px;margin-bottom:14px}
.card h2{margin:0 0 12px;font-size:13px;text-transform:uppercase;
  letter-spacing:.06em;color:var(--muted);font-weight:600}
label{display:block;margin:10px 0 4px;font-size:12px;color:var(--muted)}
input[type=text],input[type=number],select{width:100%;padding:7px 9px;
  border:1px solid var(--line);border-radius:6px;background:var(--bg);
  color:var(--ink);font:inherit}
.row{display:flex;gap:8px}.row>*{flex:1}
.check{display:flex;align-items:center;gap:7px;margin:7px 0;font-size:13px}
.check input{margin:0}
button{padding:9px 14px;border-radius:7px;border:1px solid var(--line);
  background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button.ghost{background:transparent;color:var(--ink)}
button:disabled{opacity:.45;cursor:not-allowed}
.bar{height:7px;background:var(--line);border-radius:4px;overflow:hidden;margin:8px 0}
.bar>i{display:block;height:100%;background:var(--accent);width:0;transition:width .3s}
.log{font-family:var(--mono);font-size:11.5px;background:var(--bg);
  border:1px solid var(--line);border-radius:6px;padding:9px;height:190px;
  overflow:auto;white-space:pre-wrap;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;
  letter-spacing:.04em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{border:1px solid var(--line);border-radius:8px;padding:11px 13px}
.kpi b{display:block;font-size:24px;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em}
.kpi span{font-size:11.5px;color:var(--muted)}
.delta.up{color:var(--good)}.delta.down{color:var(--bad)}
.pill{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
  border:1px solid var(--line);color:var(--muted)}
.banner{padding:9px 12px;border-radius:7px;font-size:12.5px;margin-bottom:12px;
  border:1px solid var(--warn);color:var(--warn)}
.tlist{max-height:300px;overflow:auto;border:1px solid var(--line);border-radius:7px}
.tlist div{padding:7px 10px;border-bottom:1px solid var(--line);cursor:pointer;
  font-size:12.5px;display:flex;justify-content:space-between;gap:8px}
.tlist div:hover{background:var(--bg)}
.tlist div.sel{background:var(--bg);font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;flex:none;align-self:center}
.dot.ok{background:var(--good)}.dot.no{background:var(--bad)}
.debate{display:grid;grid-template-columns:220px 1fr;gap:16px}
@media (max-width:760px){.debate{grid-template-columns:1fr}}
.debate img{width:100%;border-radius:8px;border:1px solid var(--line)}
.rnd{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:9px}
.rnd h4{margin:0 0 5px;font-size:12.5px}
.rnd .verdict{font-family:var(--mono);font-size:12px;color:var(--accent)}
.rnd p{margin:5px 0 0;font-size:12.5px;color:var(--muted)}
.tabs{display:flex;gap:4px;margin-bottom:12px;flex-wrap:wrap}
.tabs button{background:transparent;color:var(--muted);border:1px solid transparent;
  font-weight:500;padding:6px 11px}
.tabs button.on{border-color:var(--line);color:var(--ink);background:var(--bg)}
.hide{display:none}
</style></head><body>

<header><h1>VIDA + PANDA<small>다중 에이전트 작물병해 진단 · 로컬 실행 콘솔</small></h1></header>

<div class="wrap">
<!-- ── 좌측: 설정 ────────────────────────────────────────────────── -->
<div>
  <div class="card">
    <h2>실행 설정</h2>
    <label>데이터셋 폴더 (안에 "작물,병해" 폴더들)</label>
    <input type="text" id="dataset_root">
    <label>결과 저장 폴더</label>
    <input type="text" id="output_dir">
    <div class="row">
      <div><label>이미지 수</label><input type="number" id="n_images" min="1"></div>
      <div><label>토론 참가자</label><input type="number" id="n_debate" min="2"></div>
    </div>
    <div class="row">
      <div><label>시드</label><input type="number" id="seed"></div>
      <div><label>Softmax T</label><input type="number" id="softmax_temp" step="0.5"></div>
    </div>
    <div class="check"><input type="checkbox" id="mock">
      <label for="mock" style="margin:0">가짜 에이전트 (무료 · 코드 검증용)</label></div>
    <div class="check"><input type="checkbox" id="use_judge">
      <label for="use_judge" style="margin:0">GPT-4.1 심판 채점 사용</label></div>
    <div class="check"><input type="checkbox" id="require_new_evidence">
      <label for="require_new_evidence" style="margin:0">반-아첨 장치 켜기</label></div>
  </div>

  <div class="card">
    <h2>에이전트</h2>
    <div id="agentbox"><span class="pill">불러오는 중…</span></div>
    <div id="provnotes" style="font-size:11.5px;color:var(--muted);margin-top:9px"></div>
  </div>

  <div class="card">
    <h2>실행</h2>
    <div class="row">
      <button id="start">시작</button>
      <button id="stop" class="ghost" disabled>중단</button>
    </div>
    <div class="bar"><i id="bar"></i></div>
    <div style="font-size:12px;color:var(--muted)" id="phase">대기 중</div>
    <div class="log" id="log"></div>
  </div>
</div>

<!-- ── 우측: 결과 ────────────────────────────────────────────────── -->
<div>
  <div id="mockbanner" class="banner hide">
    가짜 에이전트 모드입니다 — 숫자는 무작위로 생성된 값이며 과학적 의미가 없습니다.
  </div>

  <div class="card">
    <h2>토론 효과</h2>
    <div class="kpis" id="kpis"><span class="pill">아직 실행 전</span></div>
  </div>

  <div class="card">
    <h2>VIDA 에이전트별 성적</h2>
    <div style="overflow-x:auto"><table id="metrics"><tbody>
      <tr><td style="color:var(--muted)">아직 실행 전</td></tr></tbody></table></div>
  </div>

  <div class="card">
    <h2>토론 기록</h2>
    <div class="tabs">
      <button class="on" data-tab="list">이미지 목록</button>
      <button data-tab="infl">누가 누구를 설득했나</button>
    </div>
    <div id="tab-list">
      <div class="tlist" id="tlist"><div style="color:var(--muted);cursor:default">
        아직 실행 전</div></div>
      <div id="detail" style="margin-top:14px"></div>
    </div>
    <div id="tab-infl" class="hide"><div id="infl"></div></div>
  </div>
</div>
</div>

<script>
const $ = s => document.querySelector(s);
const CFG_KEYS = ['dataset_root','output_dir','n_images','n_debate','seed','softmax_temp'];
const FLAGS = ['mock','use_judge','require_new_evidence'];
let selected = null, lastPhase = '';

// ── 초기값 ───────────────────────────────────────────────────────────
fetch('/api/defaults').then(r=>r.json()).then(d=>{
  CFG_KEYS.forEach(k=>$('#'+k).value = d[k]);
  FLAGS.forEach(k=>$('#'+k).checked = !!d[k]);
});

fetch('/api/providers').then(r=>r.json()).then(p=>{
  const box = $('#agentbox');
  const real = p.agents || [];
  let html = '';
  if(real.length){
    html += real.map(a=>`<div class="check">
      <input type="checkbox" class="ag" value="${a.name}" checked>
      <label style="margin:0">${a.name} <span class="pill">${a.provider}</span></label>
    </div>`).join('');
  } else {
    html += `<div style="font-size:12.5px;color:var(--muted)">
      사용 가능한 실제 에이전트가 없습니다.<br>
      · 무료: Ollama 설치 후 <code>ollama pull qwen2.5vl:3b</code><br>
      · 유료: .env 에 API 키 입력<br>
      우선은 <b>가짜 에이전트</b> 체크로 UI를 둘러보세요.</div>`;
  }
  html += `<div style="margin-top:10px;font-size:11.5px;color:var(--muted)">
    가짜 에이전트: ${p.mock_agents.join(', ')}</div>`;
  box.innerHTML = html;
  let diag = '';
  const E = p.env || {};
  if(E.exists){
    diag = `.env 읽음 (${E.encoding}) · 인식된 키: ` +
           (E.loaded_keys && E.loaded_keys.length ? E.loaded_keys.join(', ') : '없음');
  } else {
    diag = `.env 파일을 못 찾음 → ${E.path || '?'}`;
  }
  if(E.error) diag += ' · ' + E.error;
  if(p.notes && p.notes.length) diag += '<br>연결 메모: ' + p.notes.join(' | ');
  $('#provnotes').innerHTML = diag;
});

// ── 실행 ─────────────────────────────────────────────────────────────
$('#start').onclick = async () => {
  const cfg = {};
  CFG_KEYS.forEach(k=>cfg[k] = $('#'+k).value);
  FLAGS.forEach(k=>cfg[k] = $('#'+k).checked);
  const picked = [...document.querySelectorAll('.ag:checked')].map(x=>x.value);
  cfg.agents = cfg.mock ? null : (picked.length ? picked : null);
  const r = await fetch('/api/start',{method:'POST',body:JSON.stringify(cfg)});
  if(!r.ok){ alert((await r.json()).error); return; }
  selected = null; $('#detail').innerHTML = '';
};
$('#stop').onclick = () => fetch('/api/stop',{method:'POST',body:'{}'});

document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  $('#tab-list').classList.toggle('hide', b.dataset.tab!=='list');
  $('#tab-infl').classList.toggle('hide', b.dataset.tab!=='infl');
});

// ── 폴링 ─────────────────────────────────────────────────────────────
const fmt = v => (v===null||v===undefined||v==='') ? '—'
                : (typeof v==='number' ? v.toFixed(3) : v);
const PHASE_KO = {idle:'대기 중', VIDA:'VIDA 1라운드', Judge:'심판 채점',
  'PANDA R2/R3':'PANDA 토론', done:'완료', stopped:'중단됨', error:'오류'};

let _lastSummary = {};
const su_n = () => _lastSummary.n_images || 0;

async function poll(){
  let s;
  try { s = await (await fetch('/api/state')).json(); } catch(e){ return; }
  const running = ['VIDA','Judge','PANDA R2/R3'].includes(s.phase);
  $('#start').disabled = running; $('#stop').disabled = !running;
  const pct = s.total ? Math.round(s.done/s.total*100) : (s.phase==='done'?100:0);
  $('#bar').style.width = pct+'%';
  const ko = PHASE_KO[s.phase] || s.phase;
  $('#phase').textContent = s.phase==='idle' ? '대기 중'
      : (s.phase==='done' ? `완료 · ${su_n()}장 처리`
                          : `${ko} · ${s.done}/${s.total} (${pct}%)`)
        + (s.error ? ' · 오류: '+s.error : '');
  $('#log').textContent = (s.log||[]).join('\n');
  $('#log').scrollTop = 1e9;

  $('#mockbanner').classList.toggle('hide', !(s.summary && s.summary.mock));

  // 지표표
  if(s.metrics && s.metrics.length){
    const rq = s.metrics.some(m=>m.reasoning_quality!==null);
    $('#metrics').innerHTML =
      `<thead><tr><th>에이전트</th><th class="num">작물</th><th class="num">병해</th>
       <th class="num">종합</th><th class="num">파싱실패%</th><th class="num">확신도</th>
       ${rq?'<th class="num">추론품질</th>':''}</tr></thead><tbody>` +
      s.metrics.map(m=>`<tr><td>${m.agent}</td>
        <td class="num">${fmt(m.crop_acc)}</td>
        <td class="num">${fmt(m.disease_acc)}</td>
        <td class="num"><b>${fmt(m.combined_acc)}</b></td>
        <td class="num">${m.parse_fail_pct==null?'—':m.parse_fail_pct.toFixed(1)+'%'}</td>
        <td class="num">${fmt(m.mean_confidence)}</td>
        ${rq?`<td class="num">${fmt(m.reasoning_quality)}</td>`:''}</tr>`).join('') +
      '</tbody>';
  }

  // KPI
  const su = s.summary || {}; _lastSummary = su;
  if(su.n_images){
    const d = su.delta, cls = d>0.001?'up':(d<-0.001?'down':'');
    const arrow = d>0.001?'▲':(d<-0.001?'▼':'→');
    $('#kpis').innerHTML = `
      <div class="kpi"><b>${su.baseline_both.toFixed(3)}</b><span>토론 전 (R1 가중투표)</span></div>
      <div class="kpi"><b>${su.consensus_both.toFixed(3)}</b><span>토론 후 (R3 가중투표)</span></div>
      <div class="kpi"><b class="delta ${cls}">${arrow} ${d>=0?'+':''}${d.toFixed(3)}</b>
        <span>토론 효과 Δ (종합 정확도)</span></div>
      <div class="kpi"><b>${su.n_images}</b><span>이미지 수</span></div>`;
    const inf = su.influence || {};
    const keys = Object.keys(inf).sort((a,b)=>inf[b]-inf[a]);
    $('#infl').innerHTML = keys.length
      ? `<table><thead><tr><th>설득한 쪽 → 바뀐 쪽</th><th class="num">횟수</th></tr></thead>
         <tbody>${keys.map(k=>`<tr><td>${k.replace('→',' → ')}</td>
         <td class="num">${inf[k]}</td></tr>`).join('')}</tbody></table>
         <p style="font-size:12px;color:var(--muted);margin-top:10px">고집도(한 번도 진단을
         안 바꾼 비율): ${Object.entries(su.stubborn_pct||{})
           .map(([a,v])=>`${a} ${v}%`).join(' · ')}</p>`
      : '<p style="color:var(--muted);font-size:12.5px">서로 설득한 기록이 없습니다.</p>';
  }

  // 토론 목록
  if(s.n_transcripts && s.phase !== lastPhase || (s.n_transcripts && !$('#tlist').dataset.n)
     || (s.n_transcripts && +$('#tlist').dataset.n !== s.n_transcripts)){
    const list = await (await fetch('/api/transcripts')).json();
    $('#tlist').dataset.n = s.n_transcripts;
    $('#tlist').innerHTML = list.map(t=>`
      <div data-n="${t.name}" class="${t.name===selected?'sel':''}">
        <span class="dot ${t.ok?'ok':'no'}"></span>
        <span style="flex:1;overflow:hidden;text-overflow:ellipsis">${t.label||t.name}</span>
        <span style="color:var(--muted)">${t.consensus}</span></div>`).join('');
    $('#tlist').querySelectorAll('div[data-n]').forEach(el=>{
      el.onclick = ()=>showDetail(el.dataset.n);
    });
  }
  lastPhase = s.phase;
}

async function showDetail(name){
  selected = name;
  $('#tlist').querySelectorAll('div[data-n]').forEach(
    el=>el.classList.toggle('sel', el.dataset.n===name));
  const t = await (await fetch('/api/transcript?name='+encodeURIComponent(name))).json();
  if(!t.rounds) return;
  const V = r => `${r.crop_category||'?'} / ${r.disease_category||'?'}`;
  const body = Object.entries(t.rounds).map(([agent,r])=>`
    <div class="rnd">
      <h4>${agent}</h4>
      <div class="verdict">R1 ${V(r.r1)} → R2 ${V(r.r2)} → R3 ${V(r.r3)}</div>
      <p><b>R3 근거:</b> ${r.r3.reasoning||'—'}</p>
      <p><b>설득한 상대:</b> ${r.r3.influenced_by||'None'} ·
         <b>새 시각적 근거:</b> ${r.r3.new_visual_evidence||'NONE'} ·
         <b>확신도:</b> ${r.r3.confidence??'—'}</p>
    </div>`).join('');
  $('#detail').innerHTML = `<div class="debate">
      <div>
        <img src="/img?name=${encodeURIComponent(name)}" alt="">
        <p style="font-size:12px;margin:8px 0 0">
          <b>정답</b> ${t.gt}<br>
          <b>토론 전</b> ${t.baseline}<br>
          <b>토론 후</b> ${t.consensus}</p>
      </div>
      <div>${body}</div></div>`;
}

poll(); setInterval(poll, 1200);
</script>
</body></html>
"""


def main():
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    url = f'http://localhost:{PORT}'
    print(f"\n  VIDA + PANDA 콘솔 실행 중 →  {url}")
    print("  종료하려면 이 창에서 Ctrl+C\n")
    try:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  종료합니다.")
        srv.shutdown()


if __name__ == '__main__':
    main()
