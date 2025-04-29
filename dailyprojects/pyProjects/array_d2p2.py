 ##performing reverse of string


 ##output

 ##nums = ['d','c','b','a']


def reverse(nums):
    l = len(nums)
    n=l-1
    for i in range(l):

        if i <n:
            nums[i],nums[n] = nums[n],nums[i]

            n = n-1
    return nums

nums = ['a', 'b', 'c', 'd','e']
print(reverse(nums))


def reverse(nums):
    l = len(nums)
    n=l-1
    while i<n:
        nums[i],nums[n] = nums[n],nums[i]
        i=i+1
        n = n-1
    return nums