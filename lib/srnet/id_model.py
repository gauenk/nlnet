
class IdentityModel():

    def __init__(self):
        pass

    def __call__(self,vid,flows=None):
        return vid

    def forward(self,vid,flows=None):
        return vid

    @property
    def times(self):
        fields = ["timer_attn","timer_extract",
                  "timer_search","timer_agg","timer_fold"]
        return {field:-1 for field in fields}


