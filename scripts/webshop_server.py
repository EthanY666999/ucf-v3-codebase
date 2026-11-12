"""Loopback-only bridge for the upstream Princeton text simulator."""
import argparse
import hashlib
import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer


def full_catalog_identity(env):
    products = env.server.all_products
    indexed = int(env.server.search_engine.num_docs)
    if len(products) <= 100_000 or indexed != len(products):
        raise ValueError(f'Expected full matched catalog/index; products={len(products)}, indexed={indexed}. '
                         'Select both full paths before building the upstream index.')
    digest = hashlib.sha256()
    for product in products:
        digest.update((product['asin']+'\\n').encode())
    return {'products': len(products), 'indexed': indexed, 'product_ids_sha256': digest.hexdigest()}


class Simulator:
    def __init__(self, env, catalog_info=None):
        self.catalog_info = catalog_info or {}
        self.env = env
        goals = env.server.goals
        # Include order, attributes and goal descriptions. Only the digest leaves the process.
        identity = json.dumps([self.catalog_info,goals], sort_keys=True, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else sorted(x)).encode()
        self.catalog = hashlib.sha256(identity).hexdigest()
        self.count = len(goals)

    def health(self):
        return {'catalog': self.catalog, 'n_goals': self.count, **self.catalog_info}

    def replay(self, goal, actions):
        if type(goal) != int or not 0 <= goal < self.count:
            raise ValueError('Unknown numeric goal ID')
        if not isinstance(actions, list) or len(actions) > 15 or any(not isinstance(a, str) for a in actions):
            raise ValueError('Invalid action history')
        env = self.env
        session = str(goal)
        env.server.user_sessions.pop(session, None)
        # Replaying a prefix must reproduce the same page even for randomized search results.
        import numpy as np
        random.seed(goal)
        np.random.seed(goal)
        env.browser.current_url = None
        observation, _ = env.reset(session=goal)
        done, score = False, 0.
        try:
            for action in actions:
                if done:
                    raise ValueError('Action after a terminal purchase')
                observation, score, done, _ = env.step(action)
            return {'catalog': self.catalog, 'observation': observation,
                    'available': env.get_available_actions(), 'done': bool(done), 'score': float(score),
                    # The upstream SimServer records option name -> clicked value in
                    # item_page and uses this same dictionary to score the purchase.
                    'selected_options': dict(env.server.user_sessions[session]['options'])}
        finally:
            env.server.user_sessions.pop(session, None)


def handler_for(simulator):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.respond(200, simulator.health())
            else:
                self.respond(404, {'error': 'unknown route'})

        def do_POST(self):
            if self.path != '/replay':
                return self.respond(404, {'error': 'unknown route'})
            try:
                size = int(self.headers['Content-Length'])
                if not 0 < size <= 1_000_000:
                    raise ValueError('Invalid body size')
                req = json.loads(self.rfile.read(size))
                result = simulator.replay(req['goal_id'], req['actions'])
            except (KeyError, ValueError, TypeError) as exc:
                return self.respond(400, {'error': str(exc)})
            self.respond(200, result)

        def log_message(self, fmt, *args):
            if args and str(args[1]) != '200':
                super().log_message(fmt, *args)
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--products', help='Path to the official items JSON')
    args = p.parse_args()
    import numpy as np
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    random.seed(42)
    np.random.seed(42)
    kwargs = {'file_path': args.products} if args.products else {}
    env = WebAgentTextEnv(observation_mode='text', human_goals=True, **kwargs)
    sim = Simulator(env, full_catalog_identity(env))
    print(json.dumps(sim.health()), flush=True)
    HTTPServer(('127.0.0.1', args.port), handler_for(sim)).serve_forever()


if __name__ == '__main__':
    main()
