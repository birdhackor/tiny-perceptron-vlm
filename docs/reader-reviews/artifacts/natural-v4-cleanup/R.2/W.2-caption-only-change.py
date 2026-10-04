animals = ["鳥", "貓", "狗"]
record = {"animal": "貓", "caption": "貓在吃飯"}
print(animals[0], len(animals))
print(record["caption"])
for animal in animals:
    print("看到", animal)
assert len(animals) == 3
