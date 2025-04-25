import numpy as np

nums = [1,2,3,4,5]
k = 2

##expected output
##nums = [4,5,1,2,3]
# k=1
# nums = [5,1,2,3,4]
# 1 = nums[4]
# 2 = nums[0]
# 3 = nums[1]
# 4 = nums[2]
# 5 = nums[3]

##whats happing here in steps

##position is changing for each element
##and position of last elemt is now beocmes pos of first element
##and position from 1st element is shifting towards next position in

# new[0] = nums[4]
# new[1] = nums[0]
# new[2] = nums[1]
# new[3] = nums[2]
# new[4] = nums[3]
#
#
# k=2
# nums = [4,5,1,2,3]
# 1 = nums[3]
# 2 = nums[4]
# 3 = nums[0]
# 4 = nums[1]
# 5 = nums[2]
#
# new[0] = nums[3]
# new[1] = nums[4]
# new[2] = nums[0]
# new[3] = nums[1]
# new[4] = nums[2]
#
#
# new[:k] = nums[k-1:]

new = []
n = len(nums)
new[:k] = nums[n-k:]
new[k:] = nums[:n-k]
print(new)


##here drawbacks are you are using extra space new = [] array
##instead try rotating inplace within same array


##now inplace rotation

##reverse entire array
##reverse first k
##reverse remaing
##for inplace rotations you need to take care of elements position change happen at same time
##--otherwise you may loose the element

##nums[start],nums[end] = nums[end],nums[start]
##
#
# k=0
#
# nums[0],nums[4] = nums[4],nums[0]
# nums[5,2,3,4,1]
#
# nums[1],nums[4] = nums[4],nums[1]
# nums[5,1,3,4,2]
#
# nums[2],nums[4] = nums[4],nums[2]
# nums[5,1,2,4,3]
#
# nums[3],nums[4] = nums[4],nums[3]
# nums[5,1,2,3,4]
#
# k=1
# nums[5,1,2,3,4]
#
# nums[0],nums[3] = nums[3],nums[0]
# nums[4,1,2,3,5]
#
# nums[1],nums[3] = nums[3],nums[1]
# nums[4,3,2,1,5]
#
# nums[2],nums[3] = nums[3],nums[2]
# nums[4,3,1,2,5]
#
# k = 2
# nums[4,3,1,2,5]
#
# nums[0],nums[2] = nums[2],nums[0]




##basic reverse function
def rotate(nums,k):
    n = len(nums)
    k %= n
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1

    reverse(0,n-1)
    reverse(0,k-1)
    reverse(k,n-1)
    return nums

print(rotate(nums,k))













