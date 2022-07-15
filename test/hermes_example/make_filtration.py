
filtration = [str(0.01*i) for i in range(0,1600)]
filtration_string = " ".join(filtration)
with open("filtration2.txt","w") as fp:
    fp.write(filtration_string)

