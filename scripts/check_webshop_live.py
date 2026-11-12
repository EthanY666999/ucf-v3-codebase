"""Exercise a real training goal on the upstream simulator before launching a GPU."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from webshop import bridge_call


def check(url, goal=1500, query='bag'):
    if goal < 1500:
        raise ValueError('Use a training goal for the live check')
    def replay(history):
        return bridge_call(url,'/replay',{'goal_id':goal,'actions':history})
    initial = replay([])
    if not initial['available']['has_search_bar']:
        raise ValueError('Expected a search page')
    search = ['search['+query+']']
    results = replay(search)
    if not results['available']['clickables'] or results != replay(search):
        raise ValueError('Search yielded no visible actions or replay changed')
    products = [x for x in results['available']['clickables'] if len(x)==10 and x.isalnum()]
    if not products:
        raise ValueError('No product found; try another --query or check the search index')
    history = search+['click['+products[0]+']']
    detail = replay(history)
    if 'buy now' not in [x.lower() for x in detail['available']['clickables']]:
        raise ValueError('The clicked product did not open a purchasable item')
    purchase = replay(history+['click[buy now]'])
    assert purchase['done'] and 0 <= purchase['score'] <= 1
    assert replay([]) == initial
    return dict(goal_id=goal,catalog=initial['catalog'],replay=True,terminal=True,official_score=purchase['score'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8765')
    p.add_argument('--goal',type=int,default=1500)
    p.add_argument('--query',default='bag')
    args=p.parse_args()
    print(json.dumps(check(args.url,args.goal,args.query),indent=2))
