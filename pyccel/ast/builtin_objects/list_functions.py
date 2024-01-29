#list.append()
class AppendFunction:
    def apply(self, list_object, value):
        # Logic to handle append operation
        list_object.append(value)
        return list_object

#list.pop()
class PopFunction:
    def apply(self, list_object, index=None):
        # Logic to handle pop operation
        if index is None:
            return list_object.pop()
        else:
            return list_object.pop(index)

#list.insert()
class InsertFunction:
    def apply(self, list_object, index, value):
        # Logic to handle insert operation
        list_object.insert(index, value)
        return list_object

#list.clear()
class ClearFunction:
    def apply(self, list_object):
        # Logic to handle clear operation
        list_object.clear()
        return list_object

#list.extend()
class ExtendFunction:
    def apply(self, list_object, iterable):
        # Logic to handle extend operation
        list_object.extend(iterable)
        return list_object

#list.count()
class CountFunction:
    def apply(self, list_object, value):
        # Logic to handle count operation
        return list_object.count(value)

#list.index()
class IndexFunction:
    def apply(self, list_object, value):
        # Logic to handle index operation
        return list_object.index(value)

#list.reverse()
class ReverseFunction:
    def apply(self, list_object):
        # Logic to handle reverse operation
        list_object.reverse()
        return list_object

#list.remove()
class RemoveFunction:
    def apply(self, list_object, value):
        # Logic to handle remove operation
        list_object.remove(value)
        return list_object

#list.sort()
class SortFunction:
    def apply(self, list_object, key=None, reverse=False):
        # Logic to handle sort operation
        list_object.sort(key=key, reverse=reverse)
        return list_object

#list-slicing
class SliceFunction:
    def apply(self, list_object, start=None, stop=None, step=None):
        # Logic to handle slice operation
        sliced_list = list_object[start:stop:step]
        return sliced_list