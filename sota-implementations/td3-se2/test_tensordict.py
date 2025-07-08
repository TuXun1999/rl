import tensordict

a = 1
b = 2


td = tensordict.TensorDict()
td["a"] = a
td["b"] = b
td.set("a")
print(td)

td2 = tensordict.TensorDict()