"""
VIDA + PANDA 핵심 파이프라인 (노트북에서 추출한 재사용 모듈).

노트북과 웹 UI가 같은 코드를 쓰도록 분리했습니다.
로직은 원본 노트북과 동일하며, 진행상황 콜백과 중단 플래그만 추가됐습니다.
"""
from __future__ import annotations
import os, json, base64, random, re, math, time, threading
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════
# 1. 분류 체계
# ══════════════════════════════════════════════════════════════════════════
CDDM_CLASSES = [
    "Apple,Alternaria Blotch","Apple,Black Rot","Apple,Brown Spot",
    "Apple,Cedar Apple Rust","Apple,Frog Eye Leaf Spot","Apple,Grey Spot",
    "Apple,Healthy","Apple,Leaf Rust","Apple,Mosaic Virus",
    "Apple,Powdery Mildew","Apple,Scab",
    "Bell Pepper,Bacterial Spot","Bell Pepper,Healthy",
    "Blueberry,Healthy","Cherry,Healthy","Cherry,Powdery Mildew",
    "Corn,Healthy","Corn,Leaf Rust","Corn,Leaf Spot","Corn,Northern Leaf Blight",
    "Grape,Black Rot","Grape,Esca","Grape,Healthy","Grape,Leaf Blight",
    "Orange,Citrus Greening","Orange,Healthy",
    "Peach,Bacterial Spot","Peach,Healthy",
    "Potato,Early Blight","Potato,Healthy","Potato,Late Blight",
    "Pumpkin,Powdery Mildew","Raspberry,Healthy",
    "Rice,Bacterial Leaf Blight","Rice,Blast","Rice,Brown Spot",
    "Rice,Leaf Blight","Rice,Leaf Smut","Rice,Tungro",
    "Soybean,Healthy","Strawberry,Healthy","Strawberry,Leaf Scorch",
    "Tomato,Bacterial Spot","Tomato,Early Blight","Tomato,Healthy",
    "Tomato,Late Blight","Tomato,Leaf Mold","Tomato,Mosaic Virus",
    "Tomato,Powdery Mildew","Tomato,Septoria Leaf Spot","Tomato,Spider Mites",
    "Tomato,Target Spot","Tomato,Yellow Leaf Curl","Wheat,Healthy",
]
CDDM_CROPS    = sorted(set(c.split(',')[0].strip() for c in CDDM_CLASSES))
CDDM_DISEASES = sorted(set(c.split(',')[1].strip() for c in CDDM_CLASSES))
HEALTHY_LABEL = "Healthy"


class Taxonomy:
    """데이터셋 폴더 이름에서 읽어낸 분류체계.

    폴더가 "Citrus,Canker" 처럼 <작물>,<병해> 형태이면 그대로 분류체계가 됩니다.
    따라서 CDDM이든 감귤이든 코드를 고치지 않고 같은 파이프라인을 씁니다.
    """

    def __init__(self, classes):
        self.classes  = sorted(set(classes))
        self.crops    = sorted(set(c.split(',')[0].strip() for c in self.classes))
        self.diseases = sorted(set(c.split(',', 1)[1].strip() for c in self.classes))
        self.crop_list    = ', '.join(self.crops)
        self.disease_list = ', '.join(self.diseases)

    def __len__(self): return len(self.classes)

    @classmethod
    def from_spec(cls, spec):
        """dataset.json 명세에서 분류체계를 구성."""
        crop = spec.get('crop', 'Unknown')
        return cls([f"{crop},{d}" for d in spec['classes']])

    @classmethod
    def from_folder(cls, root):
        """<root> 아래에서 이름에 쉼표가 있는 폴더를 분류체계로 인식."""
        root = Path(root)
        found = []
        if root.exists():
            for d in sorted(root.iterdir()):
                if d.is_dir() and ',' in d.name:
                    found.append(d.name.strip())
        return cls(found) if found else None

    @classmethod
    def cddm(cls):
        return cls(CDDM_CLASSES)


TAXONOMY = Taxonomy.cddm()      # 실행 시 데이터셋에 맞춰 교체됨


# ══════════════════════════════════════════════════════════════════════════
# 2. 데이터셋 로딩 (층화 추출)
# ══════════════════════════════════════════════════════════════════════════
def load_dataset(root, n_images=10, seed=42, taxonomy=None):
    """<root>/<"작물,병해">/*.jpg 구조에서 병해80/건강20 비율로 추출.

    taxonomy 를 주지 않으면 폴더 이름에서 자동으로 만들어냅니다.
    반환: (pilot, missing, taxonomy)
    """
    root = Path(root)

    # ── dataset.json 명세 모드 ────────────────────────────────────────────
    # 원본 파일을 옮기거나 복사하지 않고, 폴더 경로만 클래스에 연결합니다.
    #   {"crop":"Citrus", "classes":{"Greening":["D:/.../greening"], ...}}
    spec = None
    if root.suffix.lower() == '.json' and root.exists():
        spec = json.loads(root.read_text(encoding='utf-8'))
        tax = taxonomy or Taxonomy.from_spec(spec)
    else:
        tax = taxonomy or Taxonomy.from_folder(root) or Taxonomy.cddm()

    rng = random.Random(seed)
    all_pairs, missing = [], []
    per_class = max(2, math.ceil(n_images / max(len(tax), 1)))

    for cl in tax.classes:
        if spec is not None:
            disease = cl.split(',', 1)[1].strip()
            dirs = [Path(x) for x in spec['classes'].get(disease, [])]
            imgs = []
            for dd in dirs:
                if dd.exists():
                    imgs += [f for f in sorted(dd.rglob('*'))
                             if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]
            if not imgs:
                missing.append(cl); continue
        else:
            d = root / cl
            if not d.exists():
                missing.append(cl); continue
            imgs = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")) + sorted(d.glob("*.jpeg"))
        if not imgs:
            missing.append(cl); continue
        crop, disease = [x.strip() for x in cl.split(',', 1)]
        for img_path in rng.sample(imgs, min(per_class, len(imgs))):
            all_pairs.append({
                "image_path": img_path, "class_label": cl,
                "gt_crop": crop, "gt_disease": disease,
                "gt_diseased": 0 if disease == HEALTHY_LABEL else 1,
            })

    rng.shuffle(all_pairs)
    diseased = [p for p in all_pairs if p["gt_diseased"] == 1]
    healthy  = [p for p in all_pairs if p["gt_diseased"] == 0]
    n_h = min(n_images // 5, len(healthy))
    n_d = min(n_images - n_h, len(diseased))
    pilot = rng.sample(diseased, n_d) + rng.sample(healthy, n_h)
    rng.shuffle(pilot)
    # 클래스가 달라도 파일명이 같을 수 있으므로 고유 id 를 부여합니다.
    # (예: Black spot/146.jpg 와 canker/146.jpg)
    for i, item in enumerate(pilot):
        item['uid'] = f"{i:04d}_{item['class_label'].replace(',', '_')}_{Path(item['image_path']).name}"
    return pilot, missing, tax


# ══════════════════════════════════════════════════════════════════════════
# 3. 프롬프트
# ══════════════════════════════════════════════════════════════════════════
def build_prompts(tax):
    """분류체계에 맞춘 R1/R2/R3 프롬프트 조각을 만들어 돌려줍니다."""
    r1 = (
        "You are a plant science researcher analyzing a crop leaf image. "
        "Identify the crop and any disease present.\n\n"
        "BEGIN YOUR RESPONSE WITH THESE FIELDS — NO TEXT BEFORE THEM:\n"
        f"Crop Category: <one of: {tax.crop_list}>\n"
        f"Disease Category: <one of: {tax.disease_list}>\n"
        "Diseased: <Yes or No>\n"
        "Reasoning: <2-3 sentences citing specific visual evidence>\n"
        "Confidence: <0.0 to 1.0>"
    )
    fmt2 = (
        "BEGIN YOUR RESPONSE WITH THESE FIELDS — NO TEXT BEFORE THEM:\n"
        f"Crop Category: <one of: {tax.crop_list}>\n"
        f"Disease Category: <one of: {tax.disease_list}>\n"
        "Diseased: <Yes or No>\n"
        "Reasoning: <address peers by name, cite visual evidence>\n"
        "Confidence: <0.0 to 1.0>\n"
        "Influenced By: <exact peer name who changed your verdict, or None>\n"
        "New Visual Evidence: <new visual feature you observed, or NONE>"
    )
    fmt3 = fmt2.replace("<address peers by name, cite visual evidence>",
                        "<final position, address peers by name>")
    return {'r1': r1, 'fmt2': fmt2, 'fmt3': fmt3}


def make_r2_prompt(my_name, own_r1, others_r1, fmt2, require_new_evidence=True):
    def s(p): return f"{p.get('crop_category','?')} / {p.get('disease_category','?')}"
    own = f"YOUR Round 1 verdict: {s(own_r1)}\n  Reasoning: {own_r1.get('reasoning','')[:200]}"
    peers = ''.join(f"\n[{pn}]: {s(p)}\n  Reasoning: {p.get('reasoning','')[:150]}\n"
                    for pn, p in others_r1.items())
    names = list(others_r1.keys())
    # 반-아첨 장치를 끄면(ablation) 이 문장이 빠집니다
    eviction = ("Only change your verdict if you identify NEW visual evidence "
                "not in your R1 reasoning. " if require_new_evidence else "")
    return (f"You are {my_name}, a plant science researcher.\n\n{own}\n\n"
            f"Peer assessments of the SAME image:\n{peers}\n"
            f"Re-examine the image. Address {' and '.join(names)} by name — "
            f"AGREE or DISAGREE with their crop and disease ID, citing specific "
            f"visual features. {eviction}\n\n" + fmt2)

def make_r3_prompt(my_name, own_r1, own_r2, others_r2, fmt3):
    def s(p): return f"{p.get('crop_category','?')} / {p.get('disease_category','?')}"
    hist = (f"YOUR history: R1={s(own_r1)} → R2={s(own_r2)}\n"
            f"  R2 Reasoning: {own_r2.get('reasoning','')[:200]}")
    peers = ''.join(f"\n[{pn}] R2: {s(p)}\n  Reasoning: {p.get('reasoning','')[:150]}\n"
                    for pn, p in others_r2.items())
    names = list(others_r2.keys())
    return (f"You are {my_name}. FINAL ROUND — commit to your verdict.\n\n{hist}\n\n"
            f"Peers' Round 2 positions:\n{peers}\n"
            f"Address {' and '.join(names)} by name. Final AGREE/DISAGREE with "
            f"visual evidence.\n\n" + fmt3)


# ══════════════════════════════════════════════════════════════════════════
# 4. 3-pass 응답 파서
# ══════════════════════════════════════════════════════════════════════════
def strip_md(text):
    text = re.sub(r'```[\w]*\n?', '', text)
    return re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text).strip()

def _match(t, pool):
    t = t.lower()
    for c in pool:
        if c.lower() == t: return c                       # pass 1: 정확 일치
    m = [c for c in pool if c.lower() in t or t in c.lower()]
    return max(m, key=len) if m else None                 # pass 2: 부분 일치

def _word_overlap(text_lower, pool):                      # pass 3: 단어 중첩
    words = set(re.findall(r'\w+', text_lower))
    best, best_score = None, 0
    for item in pool:
        iw = set(re.findall(r'\w+', item.lower()))
        sc = len(words & iw) / max(len(iw), 1)
        if sc > best_score: best, best_score = item, sc
    return best if best_score > 0 else None

_KNOWN = ['crop category:', 'disease category:', 'diseased:', 'reasoning:',
          'confidence:', 'influenced by:', 'new visual evidence:']

def parse_response(text, agent_name='', tax=None):
    tax = tax or TAXONOMY
    r = {'crop_category': None, 'disease_category': None, 'diseased': None,
         'reasoning': '', 'confidence': None, 'influenced_by': None,
         'new_visual_evidence': None, 'raw': text}
    if not text or text.startswith('ERROR'):
        return r
    clean = strip_md(text); clean_l = clean.lower()
    in_rsn, rsn_lines = False, []

    for line in clean.split('\n'):
        s = line.strip(); sl = s.lower()
        if any(sl.startswith(f) for f in _KNOWN): in_rsn = False
        if sl.startswith('crop category:'):
            v = s.split(':', 1)[1].strip()
            r['crop_category'] = _match(v, tax.crops) or _word_overlap(v.lower(), tax.crops)
        elif sl.startswith('disease category:'):
            v = s.split(':', 1)[1].strip()
            r['disease_category'] = _match(v, tax.diseases) or _word_overlap(v.lower(), tax.diseases)
        elif sl.startswith('diseased:'):
            v = s.split(':', 1)[1].strip().lower()
            if re.search(r'\byes\b', v): r['diseased'] = 1
            elif re.search(r'\bno\b', v): r['diseased'] = 0
        elif sl.startswith('reasoning:'):
            r['reasoning'] = s.split(':', 1)[1].strip(); in_rsn = True
        elif in_rsn and s and not any(sl.startswith(f) for f in _KNOWN):
            rsn_lines.append(s)
        elif sl.startswith('confidence:'):
            nums = re.findall(r'[0-9]*\.?[0-9]+', s.split(':', 1)[1])
            if nums:
                c = float(nums[0]); r['confidence'] = c / 100 if c > 1 else c
        elif sl.startswith('influenced by:'):
            v = s.split(':', 1)[1].strip()
            r['influenced_by'] = ('None' if v.lower() in
                ('none', 'n/a', 'na', 'no one', 'nobody', '') else v)
        elif sl.startswith('new visual evidence:'):
            v = s.split(':', 1)[1].strip()
            r['new_visual_evidence'] = v if v.upper() != 'NONE' else 'NONE'

    if rsn_lines:
        r['reasoning'] = (r['reasoning'] + ' ' + ' '.join(rsn_lines)).strip()
    if r['crop_category'] is None:
        r['crop_category'] = _word_overlap(clean_l, tax.crops)
    if r['disease_category'] is None:
        r['disease_category'] = _word_overlap(clean_l, tax.diseases)
    if r['diseased'] is None and r['disease_category']:
        r['diseased'] = 0 if r['disease_category'] == HEALTHY_LABEL else 1
    return r


# ══════════════════════════════════════════════════════════════════════════
# 5. 제공사별 호출
# ══════════════════════════════════════════════════════════════════════════
def encode_b64(path):
    with open(path, 'rb') as f: return base64.b64encode(f.read()).decode()

def mime(path):
    return 'image/jpeg' if str(path).lower().endswith(('jpg', 'jpeg')) else 'image/png'


class Providers:
    """키가 있는 제공사만 클라이언트를 만들고, 사용 가능한 에이전트 목록을 제공."""

    def __init__(self, env=None):
        e = env or os.environ
        self.clients, self.available = {}, {}
        self.errors = []

        def _openai_like(key, base=None, name='openai'):
            try:
                import openai
                self.clients[name] = (openai.OpenAI(api_key=key, base_url=base)
                                      if base else openai.OpenAI(api_key=key))
                self.available[name] = True
            except Exception as ex:
                self.errors.append(f"{name}: {ex}")

        if e.get('OPENAI_API_KEY'):    _openai_like(e['OPENAI_API_KEY'], None, 'openai')
        if e.get('XAI_API_KEY'):       _openai_like(e['XAI_API_KEY'], 'https://api.x.ai/v1', 'xai')
        self.gemini_models = []
        if e.get('GEMINI_API_KEY'):
            _openai_like(e['GEMINI_API_KEY'],
                         'https://generativelanguage.googleapis.com/v1beta/openai/', 'gemini')
            self.gemini_models = self._resolve_gemini_models(e)
        if e.get('ANTHROPIC_API_KEY'):
            try:
                import anthropic
                self.clients['anthropic'] = anthropic.Anthropic(api_key=e['ANTHROPIC_API_KEY'])
                self.available['anthropic'] = True
            except Exception as ex: self.errors.append(f"anthropic: {ex}")
        if e.get('TOGETHER_API_KEY'):
            try:
                from together import Together
                self.clients['together'] = Together(api_key=e['TOGETHER_API_KEY'])
                self.available['together'] = True
            except Exception as ex: self.errors.append(f"together: {ex}")

        # Ollama — 로컬 무료
        self.ollama_host   = e.get('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_models = []
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.ollama_host}/api/tags", timeout=4) as r:
                tags = json.loads(r.read())
            self.ollama_models = [m['name'] for m in tags.get('models', [])]
            if self.ollama_models:
                import openai
                self.clients['ollama'] = openai.OpenAI(
                    base_url=f"{self.ollama_host}/v1", api_key='ollama')
                self.available['ollama'] = True
        except Exception as ex:
            self.errors.append(f"ollama: {ex}")

    # ── Gemini 모델 자동 해결 ────────────────────────────────────────────
    # 모델명은 수시로 폐기됩니다(예: gemini-2.5-flash → 404).
    # 그래서 이름을 코드에 박지 않고, 키로 실제 쓸 수 있는 목록을 API에 물어봅니다.
    _probe_cache: dict = {}

    def _probe_gemini(self, model):
        """모델을 아주 짧게 한 번 호출해 실제로 쓸 수 있는지 확인.

        목록(models.list)에 이름이 있어도 키에 따라 404 가 나므로,
        목록만 믿지 않고 직접 찔러봅니다. 호출당 토큰 1개라 비용은 사실상 0.
        """
        if model in self._probe_cache:
            return self._probe_cache[model]
        try:
            self.clients['gemini'].chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': 'ok'}],
                max_tokens=1)
            self._probe_cache[model] = (True, None)
        except Exception as ex:
            msg = str(ex)
            dead = ('404' in msg or 'NOT_FOUND' in msg
                    or 'no longer available' in msg or 'not found' in msg.lower())
            # 429(한도 초과) 같은 일시적 오류는 '사용 가능'으로 봅니다
            self._probe_cache[model] = (False, msg) if dead else (True, None)
        return self._probe_cache[model]

    def _resolve_gemini_models(self, env):
        wanted = [x.strip() for x in env.get('GEMINI_MODELS', '').split(',') if x.strip()]
        live = []
        try:
            live = [m.id.replace('models/', '') for m in self.clients['gemini'].models.list()]
        except Exception as ex:
            self.errors.append(f"gemini 모델 목록 조회 실패: {ex}")

        def usable(mid):                   # 이미지 진단에 쓸 수 없는 것들 제외
            bad = ('embedding', 'aqa', 'imagen', 'veo', 'tts', 'learnlm',
                   'gemma', 'live', 'native-audio', 'image-generation')
            return mid.startswith('gemini') and not any(b in mid for b in bad)

        live = [m for m in live if usable(m)]

        # 1) .env 에 적힌 모델을 실제로 찔러본다
        keep, dead = [], []
        for m in wanted:
            ok, why = self._probe_gemini(m)
            (keep if ok else dead).append(m)
        if dead:
            self.errors.append(f"호출 불가로 제외: {', '.join(dead)}")

        # 2) 전부 죽었으면 살아있는 후보를 최신순으로 찔러가며 채운다
        if not keep and live:
            def rank(m):
                fam = 0 if ('flash' in m and 'lite' not in m) else (1 if 'pro' in m else 2)
                ver = 0.0
                for tok in m.split('-'):
                    try: ver = max(ver, float(tok))
                    except ValueError: pass
                return (fam, -ver)
            for m in sorted(live, key=rank):
                ok, _ = self._probe_gemini(m)
                if ok:
                    keep.append(m)
                if len(keep) >= 3:
                    break
            if keep:
                self.errors.append(f"자동 선정: {', '.join(keep)}")

        if not keep:
            self.errors.append("쓸 수 있는 Gemini 모델을 찾지 못했습니다.")
        return keep

    # ── 개별 호출 ────────────────────────────────────────────────────────
    def _chat_openai_style(self, client, model, image_path, prompt, retries=3, delay=5):
        b64 = encode_b64(image_path)
        for a in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': [
                        {'type': 'image_url',
                         'image_url': {'url': f'data:{mime(image_path)};base64,{b64}'}},
                        {'type': 'text', 'text': prompt}]}],
                    max_tokens=700, temperature=0.2)
                return (resp.choices[0].message.content or '').strip()
            except Exception as ex:
                if a < retries - 1: time.sleep(delay)
                else: return f"ERROR: {ex}"

    def _chat_anthropic(self, model, image_path, prompt, retries=3, delay=5):
        b64 = encode_b64(image_path); mt = mime(image_path)
        for a in range(retries):
            try:
                resp = self.clients['anthropic'].messages.create(
                    model=model, max_tokens=700,
                    system="You are a plant science researcher. Always begin your "
                           "response with the structured fields requested. Never add preamble.",
                    messages=[{'role': 'user', 'content': [
                        {'type': 'image', 'source': {'type': 'base64',
                                                     'media_type': mt, 'data': b64}},
                        {'type': 'text', 'text': prompt}]}])
                return resp.content[0].text.strip()
            except Exception as ex:
                if a < retries - 1: time.sleep(delay)
                else: return f"ERROR: {ex}"

    def list_agents(self):
        """UI에 보여줄 {에이전트이름: 제공사} 목록."""
        out = {}
        if self.available.get('openai'):
            out['GPT-4.1'] = 'openai'; out['GPT-4.1-mini'] = 'openai'
        if self.available.get('xai'):       out['Grok'] = 'xai'
        if self.available.get('gemini'):
            for m in self.gemini_models:
                out[m] = 'gemini'
        if self.available.get('anthropic'):
            out['Claude-Sonnet'] = 'anthropic'; out['Claude-Haiku'] = 'anthropic'
        if self.available.get('together'):  out['Gemma-3n-E4B'] = 'together'
        for m in self.ollama_models:        out[m] = 'ollama'
        return out

    def make_fn(self, agent_name):
        """에이전트 이름 → 실제 호출 함수."""
        prov = self.list_agents().get(agent_name)
        model_map = {
            'GPT-4.1': 'gpt-4.1', 'GPT-4.1-mini': 'gpt-4.1-mini',
            'Grok': os.getenv('XAI_MODEL', 'grok-4-fast-non-reasoning'),
            'Claude-Sonnet': 'claude-sonnet-4-5-20251001',
            'Claude-Haiku': 'claude-haiku-4-5-20251001',
            'Gemma-3n-E4B': 'google/gemma-3n-E4B-it',
        }
        model = model_map.get(agent_name, agent_name)
        if prov == 'anthropic':
            return lambda img, p: self._chat_anthropic(model, img, p)
        if prov == 'together':
            return lambda img, p: self._chat_openai_style(
                self.clients['together'], model, img, p)
        if prov in ('openai', 'xai', 'ollama', 'gemini'):
            return lambda img, p: self._chat_openai_style(
                self.clients[prov], model, img, p)
        raise ValueError(f"unknown agent {agent_name}")

    def judge_fn(self):
        """GPT-4.1 심판. OpenAI 키가 없으면 None 을 돌려주는 더미."""
        if not self.available.get('openai'):
            return lambda *a, **k: {'reasoning_quality': None, 'plausibility': None}

        def judge(gt_crop, gt_dis, reasoning, pred_crop, pred_dis):
            if not reasoning:
                return {'reasoning_quality': None, 'plausibility': None}
            prompt = (f"Ground truth: {gt_crop} / {gt_dis}\n"
                      f"Agent predicted: {pred_crop} / {pred_dis}\n"
                      f"Agent reasoning: {reasoning}\n\n"
                      "Rate this agricultural diagnosis reasoning on two dimensions (1-10 each):\n"
                      "Reasoning Quality: <1-10 — does the reasoning cite specific visual features?>\n"
                      "Plausibility: <1-10 — even if wrong, is it scientifically plausible?>\n"
                      "Respond ONLY with those two lines, no other text.")
            try:
                resp = self.clients['openai'].chat.completions.create(
                    model='gpt-4.1',
                    messages=[{'role': 'user', 'content': prompt}], max_tokens=60)
                t = resp.choices[0].message.content or ''
                rq = re.search(r'Reasoning Quality:\s*([0-9.]+)', t)
                pl = re.search(r'Plausibility:\s*([0-9.]+)', t)
                return {'reasoning_quality': float(rq.group(1)) if rq else None,
                        'plausibility': float(pl.group(1)) if pl else None}
            except Exception:
                return {'reasoning_quality': None, 'plausibility': None}
        return judge


# ── 가짜 에이전트 (무료 데모/회귀테스트용) ────────────────────────────────
class MockProviders:
    """정해진 확률로 정답/오답을 내는 가짜 에이전트. 과학적 의미 없음."""
    SKILL = {'Mock-Strong': 0.75, 'Mock-Mid': 0.55, 'Mock-Weak': 0.35,
             'Mock-Stubborn': 0.65, 'Mock-Sycophant': 0.45}

    def __init__(self, pilot, seed=7, tax=None):
        self.tax = tax or TAXONOMY
        self.gt = {Path(p['image_path']).name: (p['gt_crop'], p['gt_disease'])
                   for p in pilot}
        self.rng = random.Random(seed)
        self.errors, self.available = [], {'mock': True}
        self.ollama_models = []

    def list_agents(self):
        return {k: 'mock' for k in self.SKILL}

    def make_fn(self, name):
        def fn(image_path, prompt):
            time.sleep(0.01)
            gt_crop, gt_dis = self.gt[Path(image_path).name]
            acc = self.SKILL.get(name, 0.5)
            crop = gt_crop if self.rng.random() < min(acc + 0.2, 0.95) else self.rng.choice(self.tax.crops)
            dis  = gt_dis  if self.rng.random() < acc else self.rng.choice(self.tax.diseases)
            extra = ''
            if 'Influenced By' in prompt:
                peers = re.findall(r'\[([^\]]+)\]', prompt)
                # Sycophant 는 잘 휩쓸리고 Stubborn 은 거의 안 바뀌도록
                bias = {'Mock-Sycophant': 0.6, 'Mock-Stubborn': 0.05}.get(name, 0.25)
                inf = self.rng.choice(peers) if (peers and self.rng.random() < bias) else 'None'
                ev = 'NONE' if inf == 'None' else 'concentric lesion rings near the margin'
                extra = f"\nInfluenced By: {inf}\nNew Visual Evidence: {ev}"
            return (f"Crop Category: {crop}\nDisease Category: {dis}\n"
                    f"Diseased: {'No' if dis == 'Healthy' else 'Yes'}\n"
                    f"Reasoning: Irregular brown lesions with chlorotic halos along the "
                    f"leaf margin; venation consistent with {crop}.\n"
                    f"Confidence: {round(self.rng.uniform(0.55, 0.95), 2)}{extra}")
        return fn

    def judge_fn(self):
        return lambda *a, **k: {
            'reasoning_quality': round(self.rng.uniform(5, 9), 1),
            'plausibility':      round(self.rng.uniform(6, 9.5), 1)}


# ══════════════════════════════════════════════════════════════════════════
# 6. Softmax 가중 투표
# ══════════════════════════════════════════════════════════════════════════
def compute_softmax_weights(names, accs, temp=2.0):
    ex = {n: math.exp(accs.get(n, 0.0) / temp) for n in names}
    tot = sum(ex.values()) or 1.0
    return {n: v / tot for n, v in ex.items()}

def weighted_vote(values, weights):
    scores = {}
    for agent, val in values.items():
        if val is None: continue
        scores[val] = scores.get(val, 0.0) + weights.get(agent, 1.0 / max(len(values), 1))
    if not scores: return None, 0.0
    win = max(scores, key=scores.get)
    return win, scores[win]

def aggregate_votes(parsed, weights, prefix='r3'):
    agents = list(parsed)
    crop, crop_ag = weighted_vote({a: parsed[a]['crop_category'] for a in agents}, weights)
    dis,  dis_ag  = weighted_vote({a: parsed[a]['disease_category'] for a in agents}, weights)
    dbin, _       = weighted_vote({a: parsed[a]['diseased'] for a in agents}, weights)
    confs = [parsed[a]['confidence'] for a in agents if parsed[a]['confidence'] is not None]
    return {f'{prefix}_consensus_crop': crop, f'{prefix}_consensus_disease': dis,
            f'{prefix}_consensus_diseased': dbin,
            f'{prefix}_mean_confidence': (sum(confs) / len(confs)) if confs else None,
            f'{prefix}_crop_agreement': round(crop_ag, 3),
            f'{prefix}_disease_agreement': round(dis_ag, 3)}


# ══════════════════════════════════════════════════════════════════════════
# 7. 전체 실행
# ══════════════════════════════════════════════════════════════════════════
class Runner:
    """VIDA → 심판 → PANDA 를 순서대로 돌리며 진행상황을 콜백으로 흘려보냄."""

    def __init__(self, cfg, on_event=None):
        self.cfg = cfg
        self.on_event = on_event or (lambda **kw: None)
        self.stop_flag = threading.Event()
        self.out = Path(cfg['output_dir']); self.out.mkdir(parents=True, exist_ok=True)
        self.state = {'phase': 'idle', 'done': 0, 'total': 0, 'log': [],
                      'vida_rows': [], 'panda_rows': [], 'metrics': [],
                      'transcripts': {}, 'summary': {}, 'error': None}

    def log(self, msg):
        self.state['log'].append(msg)
        self.state['log'] = self.state['log'][-400:]
        self.on_event(**self.state)

    def _tick(self, phase, done, total):
        self.state.update(phase=phase, done=done, total=total)
        self.on_event(**self.state)

    # ── 메인 ─────────────────────────────────────────────────────────────
    def run(self):
        try:
            self._run()
        except Exception as ex:
            import traceback
            self.state['error'] = f"{type(ex).__name__}: {ex}"
            self.log("‼ " + self.state['error'])
            self.log(traceback.format_exc()[-1500:])
            self.state['phase'] = 'error'
            self.on_event(**self.state)

    def _run(self):
        cfg = self.cfg
        pilot, missing, tax = load_dataset(cfg['dataset_root'], cfg['n_images'], cfg['seed'])
        self.tax = tax
        P = build_prompts(tax)
        if not pilot:
            raise RuntimeError(
                f"이미지를 못 찾았습니다. 경로와 폴더 구조를 확인하세요: {cfg['dataset_root']}")
        self.pilot = pilot
        self.state['taxonomy'] = {'classes': tax.classes, 'crops': tax.crops,
                                  'diseases': tax.diseases}
        self.log(f"분류체계 자동 인식: 클래스 {len(tax)}개 / 작물 {len(tax.crops)}종 "
                 f"({', '.join(tax.crops[:6])}{'…' if len(tax.crops) > 6 else ''})")
        self.log(f"데이터셋 {len(pilot)}장 로드 "
                 f"(병해 {sum(p['gt_diseased'] for p in pilot)} / "
                 f"건강 {sum(1-p['gt_diseased'] for p in pilot)}, "
                 f"{len(set(p['class_label'] for p in pilot))}개 클래스)")
        if missing:
            self.log(f"※ 데이터 없는 클래스 {len(missing)}개는 제외")

        prov = MockProviders(pilot, tax=tax) if cfg['mock'] else Providers()
        if cfg['mock']:
            self.log("⚠ 가짜 에이전트 모드 — 결과는 과학적 의미가 없습니다 (코드 검증용)")
        for note in getattr(prov, 'errors', [])[:6]:
            if 'ollama' not in note.lower():
                self.log(f"※ {note}")
        agents = cfg['agents'] or list(prov.list_agents())
        if len(agents) < 2:
            raise RuntimeError("에이전트가 2개 이상이어야 토론이 성립합니다.")
        fns = {a: prov.make_fn(a) for a in agents}
        judge = prov.judge_fn()
        self.log(f"VIDA 에이전트 {len(agents)}개: {', '.join(agents)}")

        # ── VIDA Round 1 ────────────────────────────────────────────────
        vida, raw_bank = [], {}
        reported_err = set()          # 에이전트별 API 오류는 한 번만 보고
        total = len(pilot) * len(agents)
        k = 0
        for s_ in pilot:
            raw_bank[s_['uid']] = {}
        # R1 재사용 (ablation 용): 같은 seed·장수로 이전 실행의 R1 원문을 그대로 쓰면
        # 두 조건의 차이가 R2/R3(토론) 에서만 생깁니다 → 짝지은 비교가 가능.
        cache = {}
        if cfg.get('vida_cache'):
            cache = json.loads(Path(cfg['vida_cache']).read_text(encoding='utf-8'))
            hit = sum(1 for s_ in pilot if s_['uid'] in cache)
            self.log(f"R1 재사용: {cfg['vida_cache']} ({hit}/{len(pilot)}장 일치)")
            if hit != len(pilot):
                raise RuntimeError("R1 캐시의 이미지 목록이 다릅니다 (seed·장수·데이터셋을 같게 하세요)")

        # 에이전트를 바깥 루프로 둡니다. 로컬(Ollama) 모델은 호출할 때마다
        # VRAM 에 올라가므로, 번갈아 부르면 이미지마다 모델 교체가 일어납니다.
        # 한 모델로 전체 이미지를 끝내고 다음 모델로 넘어가면 교체가 1회로 줄어듭니다.
        for a in agents:
            if self.stop_flag.is_set(): return self._stopped()
            self.log(f"[VIDA] {a} — {len(pilot)}장 진단 시작")
            for sample in pilot:
                if self.stop_flag.is_set(): return self._stopped()
                uid  = sample['uid']
                name = Path(sample['image_path']).name
                raw = cache.get(uid, {}).get(a) if cache else None
                if raw is None:
                    raw = fns[a](sample['image_path'], P['r1'])
                if isinstance(raw, str) and raw.startswith('ERROR') and a not in reported_err:
                    reported_err.add(a)
                    self.log(f"‼ [{a}] API 오류 — {raw[:220]}")
                p = parse_response(raw, a, tax)
                raw_bank[uid][a] = raw
                cc = int(p['crop_category'] == sample['gt_crop']) if p['crop_category'] else 0
                dc = int(p['disease_category'] == sample['gt_disease']) if p['disease_category'] else 0
                vida.append({'image_name': name, 'class_label': sample['class_label'],
                             'gt_crop': sample['gt_crop'], 'gt_disease': sample['gt_disease'],
                             'agent': a, 'pred_crop': p['crop_category'],
                             'pred_disease': p['disease_category'],
                             'reasoning': p['reasoning'], 'confidence': p['confidence'],
                             'crop_correct': cc, 'disease_correct': dc,
                             'both_correct': int(cc and dc),
                             'parse_failed': int(p['crop_category'] is None
                                                 or p['disease_category'] is None)})
                k += 1
                self._tick('VIDA', k, total)
            done_a = [v for v in vida if v['agent'] == a]
            acc = sum(v['both_correct'] for v in done_a) / max(len(done_a), 1)
            self.log(f"[VIDA] {a} 완료 — 종합 정확도 {acc:.3f}")
        self.state['vida_rows'] = vida
        (self.out / 'vida_raw_bank.json').write_text(
            json.dumps(raw_bank, ensure_ascii=False, indent=1), encoding='utf-8')

        # ── 심판 ────────────────────────────────────────────────────────
        if cfg['use_judge']:
            self.log("심판 채점 중…")
            for i, row in enumerate(vida):
                if self.stop_flag.is_set(): return self._stopped()
                sc = judge(row['gt_crop'], row['gt_disease'], row['reasoning'],
                           row['pred_crop'], row['pred_disease'])
                row.update(sc)
                self._tick('Judge', i + 1, len(vida))
        else:
            for row in vida: row.update(reasoning_quality=None, plausibility=None)

        # ── 지표 ────────────────────────────────────────────────────────
        metrics = []
        for a in agents:
            g = [r for r in vida if r['agent'] == a]
            n = max(len(g), 1)
            def avg(key):
                vals = [r[key] for r in g if r.get(key) is not None]
                return round(sum(vals) / len(vals), 3) if vals else None
            metrics.append({
                'agent': a, 'n': len(g),
                'crop_acc': round(sum(r['crop_correct'] for r in g) / n, 3),
                'disease_acc': round(sum(r['disease_correct'] for r in g) / n, 3),
                'combined_acc': round(sum(r['both_correct'] for r in g) / n, 3),
                'parse_fail_pct': round(sum(r['parse_failed'] for r in g) / n * 100, 1),
                'mean_confidence': avg('confidence'),
                'reasoning_quality': avg('reasoning_quality'),
                'plausibility': avg('plausibility')})
        metrics.sort(key=lambda m: -m['combined_acc'])
        for m in metrics:
            if m['parse_fail_pct'] >= 50:
                self.log(f"⚠ [{m['agent']}] 파싱 실패 {m['parse_fail_pct']}% — "
                         f"모델명 오류이거나 응답 형식이 어긋났을 가능성이 큽니다. "
                         f"vida_raw_bank.json 의 원문을 확인하세요.")
        self.state['metrics'] = metrics
        self.log("VIDA 완료 — " + ", ".join(
            f"{m['agent']}={m['combined_acc']:.2f}" for m in metrics))

        # ── 토론 참가자 선발 + 가중치 ──────────────────────────────────
        debate = (cfg['debate_agents'] or
                  [m['agent'] for m in metrics][:cfg['n_debate']])
        accs = {m['agent']: m['combined_acc'] for m in metrics}
        W = compute_softmax_weights(debate, accs, cfg['softmax_temp'])
        self.state['weights'] = W
        self.log(f"PANDA 참가자: {', '.join(debate)} | "
                 f"가중치 " + ", ".join(f"{n}={w:.3f}" for n, w in W.items()))

        # ── PANDA R2/R3 ─────────────────────────────────────────────────
        panda, transcripts = [], {}
        total = len(pilot) * len(debate) * 2
        k = 0
        # R1 은 이미 있으므로 파싱만.
        R1 = {}
        for sample in pilot:
            uid = sample['uid']
            R1[uid] = {a: parse_response(raw_bank[uid].get(a, ''), a, tax) for a in debate}

        # R2, R3 도 에이전트를 바깥 루프로 두어 모델 교체를 라운드당 1회로 줄입니다.
        # (R2 는 전원의 R1 만, R3 는 전원의 R2 만 참조하므로 순서를 바꿔도
        #  각 에이전트가 보는 정보는 원래와 완전히 동일합니다.)
        R2, R3 = {n: {} for n in R1}, {n: {} for n in R1}

        for rnd, src, dst in (('R2', R1, R2), ('R3', R2, R3)):
            for a in debate:
                if self.stop_flag.is_set(): return self._stopped()
                self.log(f"[PANDA {rnd}] {a} — {len(pilot)}장")
                for sample in pilot:
                    if self.stop_flag.is_set(): return self._stopped()
                    uid = sample['uid']
                    others = {x: y for x, y in src[uid].items() if x != a}
                    if rnd == 'R2':
                        prompt = make_r2_prompt(a, R1[uid][a], others, P['fmt2'],
                                                cfg['require_new_evidence'])
                    else:
                        prompt = make_r3_prompt(a, R1[uid][a], R2[uid][a], others, P['fmt3'])
                    raw = fns[a](sample['image_path'], prompt)
                    parsed = parse_response(raw, a, tax); parsed['raw'] = raw
                    parsed['reverted'] = 0
                    # 반-아첨 '강제' 모드: 새 시각 근거 없이 의견을 바꾸면 코드가 되돌림.
                    # (프롬프트 지시만으로는 작은 모델이 규칙을 지키지 않음 — 1차 실험에서
                    #  의견 변경의 86% 가 새 근거 NONE 이었음)
                    if cfg.get('enforce_evidence'):
                        prev = (R1 if rnd == 'R2' else R2)[uid][a]
                        ev = str(parsed.get('new_visual_evidence') or '').strip().strip('.').upper()
                        changed = (parsed.get('disease_category') and prev.get('disease_category')
                                   and parsed['disease_category'] != prev['disease_category'])
                        if changed and ev in ('', 'NONE', 'N/A', 'NA', 'NO'):
                            for key in ('crop_category', 'disease_category', 'diseased'):
                                parsed[key] = prev.get(key)
                            parsed['reverted'] = 1
                    dst[uid][a] = parsed
                    k += 1; self._tick('PANDA R2/R3', k, total)

        # ── 라운드가 모두 끝난 뒤 이미지별로 집계 ────────────────────────
        for sample in pilot:
            uid  = sample['uid']
            name = Path(sample['image_path']).name
            r1, r2, r3 = R1[uid], R2[uid], R3[uid]

            post = aggregate_votes(r3, W, 'r3')
            pre  = aggregate_votes(r1, W, 'r1')
            row = {'image_name': name, 'class_label': sample['class_label'],
                   'gt_crop': sample['gt_crop'], 'gt_disease': sample['gt_disease'],
                   **post, **pre}
            row['baseline_crop_correct']    = int(pre['r1_consensus_crop'] == sample['gt_crop'])
            row['baseline_disease_correct'] = int(pre['r1_consensus_disease'] == sample['gt_disease'])
            row['baseline_both_correct']    = int(row['baseline_crop_correct'] and row['baseline_disease_correct'])
            row['consensus_crop_correct']    = int(post['r3_consensus_crop'] == sample['gt_crop'])
            row['consensus_disease_correct'] = int(post['r3_consensus_disease'] == sample['gt_disease'])
            row['consensus_both_correct']    = int(row['consensus_crop_correct'] and row['consensus_disease_correct'])
            for a in debate:
                for rnd, d in (('r1', r1), ('r2', r2), ('r3', r3)):
                    row[f'{rnd}_{a}_crop']    = d[a]['crop_category']
                    row[f'{rnd}_{a}_disease'] = d[a]['disease_category']
                    row[f'{rnd}_{a}_both_correct'] = int(
                        d[a]['crop_category'] == sample['gt_crop'] and
                        d[a]['disease_category'] == sample['gt_disease'])
                row[f'r2_{a}_reverted'] = r2[a].get('reverted', 0)
                row[f'r3_{a}_reverted'] = r3[a].get('reverted', 0)
                row[f'r2_{a}_influenced_by'] = r2[a]['influenced_by']
                row[f'r3_{a}_influenced_by'] = r3[a]['influenced_by']
                row[f'r2_{a}_disease_changed'] = int(
                    bool(r1[a]['disease_category']) and bool(r2[a]['disease_category'])
                    and r1[a]['disease_category'] != r2[a]['disease_category'])
                row[f'r3_{a}_disease_changed'] = int(
                    bool(r2[a]['disease_category']) and bool(r3[a]['disease_category'])
                    and r2[a]['disease_category'] != r3[a]['disease_category'])
            panda.append(row)

            transcripts[uid] = {
                'image_path': str(sample['image_path']),
                'file_name': name,
                'gt': f"{sample['gt_crop']} / {sample['gt_disease']}",
                'consensus': f"{post['r3_consensus_crop']} / {post['r3_consensus_disease']}",
                'baseline':  f"{pre['r1_consensus_crop']} / {pre['r1_consensus_disease']}",
                'rounds': {a: {
                    'r1': {kk: r1[a][kk] for kk in
                           ('crop_category', 'disease_category', 'reasoning', 'confidence')},
                    'r2': {kk: r2[a][kk] for kk in
                           ('crop_category', 'disease_category', 'reasoning', 'confidence',
                            'influenced_by', 'new_visual_evidence')},
                    'r3': {kk: r3[a][kk] for kk in
                           ('crop_category', 'disease_category', 'reasoning', 'confidence',
                            'influenced_by', 'new_visual_evidence')},
                } for a in debate}}
            self.state['panda_rows'] = panda
            self.state['transcripts'] = transcripts

        # ── 요약 + 영향 분석 ────────────────────────────────────────────
        n = max(len(panda), 1)
        base = sum(r['baseline_both_correct'] for r in panda) / n
        cons = sum(r['consensus_both_correct'] for r in panda) / n
        influence = defaultdict(int); stubborn = {a: 0 for a in debate}
        for r in panda:
            for a in debate:
                if not r[f'r2_{a}_disease_changed'] and not r[f'r3_{a}_disease_changed']:
                    stubborn[a] += 1
                for rnd in ('r2', 'r3'):
                    inf = r.get(f'{rnd}_{a}_influenced_by')
                    if inf and inf != 'None':
                        for cand in debate:
                            if cand.lower() in str(inf).lower():
                                influence[f"{cand}→{a}"] += 1; break
        self.state['summary'] = {
            'n_images': len(panda), 'agents': agents, 'debate_agents': debate,
            'weights': W, 'baseline_both': round(base, 3),
            'consensus_both': round(cons, 3), 'delta': round(cons - base, 3),
            'baseline_crop': round(sum(r['baseline_crop_correct'] for r in panda) / n, 3),
            'consensus_crop': round(sum(r['consensus_crop_correct'] for r in panda) / n, 3),
            'baseline_disease': round(sum(r['baseline_disease_correct'] for r in panda) / n, 3),
            'consensus_disease': round(sum(r['consensus_disease_correct'] for r in panda) / n, 3),
            'influence': dict(influence),
            'stubborn_pct': {a: round(stubborn[a] / n * 100, 1) for a in debate},
            'mock': cfg['mock'],
            'require_new_evidence': cfg['require_new_evidence'],
            'enforce_evidence': bool(cfg.get('enforce_evidence')),
            'reverted_changes': sum(R[u][a].get('reverted', 0) for R in (R2, R3)
                                    for u in R for a in R[u]),
        }
        self._save_csv()
        (self.out / 'panda_transcripts.json').write_text(
            json.dumps(transcripts, ensure_ascii=False, indent=1), encoding='utf-8')
        (self.out / 'summary.json').write_text(
            json.dumps(self.state['summary'], ensure_ascii=False, indent=1), encoding='utf-8')
        self.state['phase'] = 'done'
        d = self.state['summary']['delta']
        self.log(f"완료 — 토론 전 {base:.3f} → 토론 후 {cons:.3f} (Δ={d:+.3f})")
        self.on_event(**self.state)

    def _stopped(self):
        self.state['phase'] = 'stopped'
        self.log("사용자가 중단했습니다. 여기까지의 결과는 저장됩니다.")
        self._save_csv()
        self.on_event(**self.state)

    def _save_csv(self):
        import csv
        for fname, rows in (('vida_r1_results.csv', self.state['vida_rows']),
                            ('panda_results.csv',   self.state['panda_rows']),
                            ('vida_metrics.csv',    self.state['metrics'])):
            if not rows: continue
            keys = list(rows[0].keys())
            with open(self.out / fname, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
                w.writeheader(); w.writerows(rows)


DEFAULT_CFG = {
    'dataset_root': './data/images', 'output_dir': './vida_panda_results',
    'n_images': 10, 'n_debate': 3, 'seed': 42, 'softmax_temp': 2.0,
    'mock': False, 'use_judge': True, 'require_new_evidence': True,
    'agents': None, 'debate_agents': None, 'vida_cache': None,
    'enforce_evidence': False,
}
