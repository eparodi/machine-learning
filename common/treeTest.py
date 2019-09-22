from common.utils.tree import Tree

t = Tree("p1", [Tree("ch1",
                    (Tree("ch2", [Tree("ch3"), Tree("ch4")]))
                    )
         ,Tree("asd", Tree("tutu"))])

print(t)