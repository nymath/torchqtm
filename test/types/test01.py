import sys
sys.path.append('/Users/nymath/pymodules/nytrader/torchqtm')
import torchqtm.types as types


def test(data: types.BASE_FIELDS):
    print(data)
    
if __name__ == '__main__':
    test('open')
    