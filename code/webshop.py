"""Text-only interface to the local Princeton WebShop simulator."""
import hashlib
import json
import re
from types import SimpleNamespace
from urllib.request import Request, urlopen

COMMAND = re.compile(r'(search|click)\[([^\n`]+)\]', re.IGNORECASE)
BLOCK = re.compile(r'```\s*((?:search|click)\[[^\n`]+\])\s*```', re.IGNORECASE)


def command_block(text, require_argument=False):
    for block in BLOCK.finditer(text):
        if not require_argument or COMMAND.fullmatch(block.group(1)).group(2).strip():
            return block
    return None


def parse_command(text, require_argument=False):
    match = command_block(text, require_argument)
    if match and COMMAND.fullmatch(match.group(1)).group(2).strip():
        return match.group(1)
    return 0


def bridge_call(url, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url.rstrip('/') + route, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


class WebShopAgent:
    def __init__(self, require_argument=False):
        self.require_argument = require_argument
        instruction = ('Complete the shopping instruction in the simulator. Read the page and its available '
                       'actions. To search, give a query. To click, use a visible label. End your response '
                       'with one command, for example ```search[red running shoes]``` or ```click[buy now]```.')
        self.chat_completions = [{'role': 'system', 'content': instruction}]

    def update_from_env(self, observation, reward, done, info):
        self.chat_completions.append({'role': 'user', 'content': observation})

    def update_from_model(self, text):
        self.chat_completions.append({'role': 'assistant', 'content': text})
        return SimpleNamespace(action=parse_command(text, self.require_argument))


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
