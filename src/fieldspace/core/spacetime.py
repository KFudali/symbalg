from algebra.space import Space, TDomain



class SpaceTime(Space[TDomain]):
    def __init__(self, domain: TDomain, time: TimeSeries):
        super().__init__(domain)

    @property
    def time(self) -> TimeSeries
