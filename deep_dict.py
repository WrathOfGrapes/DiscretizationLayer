from pprint import pprint


class DeepDict(dict):

    @staticmethod
    def _merge(a:dict, b : dict, path=None):
            "merges b into a"
            if path is None: path = []
            for key in b:
                if key in a:
                    if isinstance(a[key], dict) and isinstance(b[key], dict):
                        DeepDict._merge(a[key], b[key], path + [str(key)])
                    else:
                        a[key] = b[key]  # same leaf value
                    # else:
                    #     raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
                else:
                    a[key] = b[key]
            return a

    def merge(self, update_dict:dict):
        return DeepDict._merge(self, update_dict)



# a = DeepDict(
#     {'russia':
#         {"moscow":5,
#         "piter": 6},
#      "usa":
#         {"ny": 1,
#         "washington": 2},
#
#      })
#
#
# b = {'russia': {"piter": 7},
#      "usa": {"ny": 1, "washington": 2, 'arisona':100}}
#
# print('a')
# pprint(a)
#
# print('b')
# pprint(b)
#
# DeepDict._merge(a, b)
# print()
# print('a')
# pprint(a)
#
# print('b')
# pprint(b)