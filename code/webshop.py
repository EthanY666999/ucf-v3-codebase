"""Text-only interface to the local Princeton WebShop simulator."""
import hashlib
import json
import re
from types import SimpleNamespace
from urllib.request import Request, urlopen

COMMAND = re.compile(r'(search|click)\[([^\n`]+)\]', re.IGNORECASE)
BLOCK = re.compile(r'```\s*((?:search|click)\[[^\n`]+\])\s*```', re.IGNORECASE)
CLASSES = {0: 'RESIDUAL', 1: 'SEARCH', 2: 'CLICK'}


def action_family(action):
    match = COMMAND.fullmatch(str(action))
    if not match or not match.group(2).strip():
        return 0
    return 1 if match.group(1).lower() == 'search' else 2


def family_tokens(tokenizer):
    """Partition by the immediate sampled token. All unlisted IDs are residual."""
    mapping = {}
    for word, family in [('search', 1), ('click', 2)]:
        for space in ('', ' '):
            text = '```' + space + word + '[x]```'
            ids = tokenizer.encode(text, add_special_tokens=False)
            prefix = ''
            for token in ids:
                piece = tokenizer.decode([token])
                if prefix.startswith('```') and not prefix[3:].strip() and piece.strip().startswith(word):
                    if token in mapping and mapping[token] != family:
                        raise ValueError('Ambiguous action token')
                    mapping[token] = family
                    break
                prefix += piece
            else:
                raise ValueError(f'Unsupported command tokenization for {text!r}')
    return mapping


def bridge_call(url, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url.rstrip('/') + route, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


class WebShopAgent:
    def __init__(self):
        instruction = ('Complete the shopping instruction in the simulator. Read the page and its available '
                       'actions. To search, give a query. To click, use a visible label. End your response '
                       'with one command, for example ```search[red running shoes]``` or ```click[buy now]```.')
        self.chat_completions = [{'role': 'system', 'content': instruction}]

    def update_from_env(self, observation, reward, done, info):
        self.chat_completions.append({'role': 'user', 'content': observation})

    def update_from_model(self, text):
        self.chat_completions.append({'role': 'assistant', 'content': text})
        match = BLOCK.search(text)
        return SimpleNamespace(action=match.group(1) if match and action_family(match.group(1)) else 0)


class WebShopEnv:
    environment = 'webshop'

    def __init__(self, task, cfg):
        self.url = cfg['webshop_url']
        self.catalog = cfg['webshop_catalog']
        self.goal = task['goal_id']
        self.history = []
        self.page = self._fetch([])
        self.raw_score = 0.

    def _fetch(self, history):
        result = bridge_call(self.url, '/replay', {'goal_id': self.goal, 'actions': history})
        if result['catalog'] != self.catalog:
            raise RuntimeError('WebShop catalog changed since the split was prepared')
        return result

    def render(self):
        choices = self.page['available']
        return (self.page['observation'] + '\nSearch available: ' + str(choices['has_search_bar'])
                + '\nClickable labels: ' + json.dumps(choices['clickables'], ensure_ascii=False))

    def anchor(self):
        return hashlib.sha256(self.render().encode()).hexdigest()

    @property
    def s(self):
        return int(self.anchor()[:16], 16)

    def success(self):
        return self.page['done'] and self.raw_score == 1.

    def resolve(self, action):
        match = COMMAND.fullmatch(str(action))
        if not match:
            return 0
        name, arg = match.group(1).lower(), match.group(2).strip()
        available = self.page['available']
        if name == 'search' and arg and available['has_search_bar']:
            return f'search[{arg}]'
        if name == 'click' and arg.lower() != 'search' and arg.lower() in {
                str(label).lower() for label in available['clickables']}:
            return f'click[{arg.lower()}]'
        return 0

    def step(self, action):
        action = self.resolve(action)
        if action:
            self.history.append(action)
            self.page = self._fetch(self.history)
        self.raw_score = float(self.page['score'])
        return self.render(), float(self.success()), self.page['done'], {'webshop_score': self.raw_score}
