# -*- coding: utf-8 -*-
minAQI_limit =[[0,41,81,381,801,1600],    #so2
               [0,41,81,181,281,400],     #no2
               #[0,51,101,251,351,430],    
               [0,31,61,91,121,250]]      #pm2.5
#maxAQI_limit =[[40,80,380,800,1599],
#               [40,80,180,280,399],
#               [50,100,250,350,429],
#               [30,60,90,120,249]]
aqi_limit = [0,51,101,201,301,401,501]
    
def get_aqi(so2,no2,pm25) :
    result = []
    result.append(aqi(so2,1))
    result.append(aqi(no2,2))
    result.append(aqi(pm25,3))
    #result.append(aqi(pm10,4))
    print(result)
    return max(result)
def aqi(conc,num)  :
    maxl = 0
    minl = 0
    minaq = 0
    maxaq = 0
    for  i in range (0,6):
        if minAQI_limit[num-1][i] >= conc :
            minaq = aqi_limit[i-1]
            maxaq = aqi_limit[i]-1
            maxl = minAQI_limit[num-1][i]-1
            minl = minAQI_limit[num-1][i-1]
            break
    #print(minaq[num-1])
    #print(maxaq[num-1])
    #print(minl)
    #print(maxl )
    aqi = minaq +(((conc-minl) * (maxaq - minaq ))/(maxl-minl))
   
    return aqi
#print(get_aqi(3,34,58))
