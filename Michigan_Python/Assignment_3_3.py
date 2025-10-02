score = input("Enter Score: ")

score_float = float(score)

if score_float < 0 or score_float >1:
    print("out of range")
elif score_float >= 0.9:
    print("A")
elif score_float >= 0.8:
    print("B")
elif score_float >= 0.7:
    print("C")
elif score_float >= 0.6:
    print("D")
else:
    print("F")