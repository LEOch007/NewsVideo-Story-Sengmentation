# meta class
class Sentences(object):
    __bg = -10      # begin timestamp
    __ed = -10      # end timestamp
    __onebest = ''  # text content

    __key  = -1        # whether keyword
    __tinterval = -1   # time interval
    __simscore = -8.0  # similarity score
    __deepscore = -8.0 # deep score

    # getting private member variables
    def getbg(self): return self.__bg
    def geted(self): return self.__ed
    def getonebest(self): return self.__onebest
    def getkey(self): return self.__key
    def gettinterval(self): return self.__tinterval
    def getsimscore(self): return self.__simscore
    def getdeepscore(self): return self.__deepscore

    # setting private member variables
    def setbg(self, sbg): self.__bg = sbg
    def seted(self, sed): self.__ed = sed
    def setonebest(self,sonebest): self.__onebest = sonebest
    def settinterval(self, tspan): self.__tinterval = tspan
    def setsimscore(self, sim): self.__simscore = sim
    def setdeepscore(self, deep): self.__deepscore = deep

    # extract features: keyword
    def keyword(self):
        slist = ['国际快讯','联播快讯','详细报道','详细内容']
        for sword in slist:
            if sword in self.__onebest:
                self.__key = 1
                return 0
        self.__key = 0
        return 0