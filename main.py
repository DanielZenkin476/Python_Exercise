from itertools import filterfalse
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        st = s.strip()
        if st == "" : return 0
        st = s.lstrip()  # remove whitespaces if present
        curr_char = st[0]
        curr_val = ord(curr_char)
        sign = 1
        leading_zero = False
        res = 0
        if curr_val == 45 or curr_val == 43: # 44-43 = 1 for plus , 44-45 = -1 for minus, the read next
            sign = 44-curr_val
            st = st[1:]
            if st == "": return 0
            curr_char = st[0]
            curr_val = ord(curr_char)
        while(48<=curr_val<=57):
            if curr_val ==48 and leading_zero == False:
                pass
            else:
                leading_zero = True
                curr_int = int(curr_char)
                res = res*10 + curr_int
            st = st[1:]
            if len(st) == 0: break
            curr_char = st[0]
            curr_val = ord(curr_char)
            if res*sign < -2**(31):
                res = 2 **(31)
                break
            if res*sign >(2 ** 31 - 1):
                res = 2 **31 -1
                break
        res = res*sign
        if res  < -2 ** (31):
            res = -2 ** (31)
        if res >(2 ** 31 - 1):
            res = 2 **31 -1
        return (res)

    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0 : return False # always not true if negative
        if x == 0 : return True
        # reverse int, chek if coimp is 0
        for_num= x
        num = 0
        while (for_num!= 0):
            num *= 10
            num+= for_num%10
            for_num = for_num // 10
        if num-x == 0:
            return True
        else : return False

    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        if n == 1 or n == 0:
            return 0
        max_mass = 0
        for i in range(0,n):
            if max_mass > height[i]* (n-i-1): pass
            else:
                for j in range(i+1,n):
                    if max_mass > height[i] * (n-i-1): break
                    max_mass = max(max_mass,min(height[i],height[j])*(j-i))
        return  max_mass

    def maxArea2(self,height):
        """
                :type height: List[int]
                :rtype: int
                """
        n = len(height)
        if n == 1 or n == 0:
            return 0
        i = 0
        j = n-1
        max_area = 0
        while (i <j):
            min_piller = min(height[i],height[j])
            max_area = max(max_area,min_piller* (j-i) )
            if min_piller == height[i]: i+=1
            else: j-=1
        return max_area

    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        roman_dict = ["I","V","X","L","C","D","M"]
        roman_val = [1,5,10,50,100,500,1000]
        #            0 1  2  3  4   5   6 even need to check, odd not
        st = ''
        i = len(roman_val)-1
        while(num>0):
            if num>= roman_val[i] or i == 0:
                st+= roman_dict[i]
                num-= roman_val[i]
            else:
                if i%2 == 0 and i != 0 :
                    if num >= roman_val[i]-roman_val[i-2] : #for exemp num >= 1000-100 = 900
                        st+= roman_dict[i-2]+roman_dict[i]
                        num-= roman_val[i]-roman_val[i-2]
                if i%2 == 1 :
                    if num >= roman_val[i]-roman_val[i-1] : #for exemp num >= 500-100 = 900
                        st+= roman_dict[i-1]+roman_dict[i]
                        num-= roman_val[i]-roman_val[i-1]
                i-=1
        return(st)

    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        i = len(s)-1
        curr_sym = ""
        roman_dict = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        flags = [False,False,False]# I,X,C
        res = 0
        while(i>=0):
            curr_sym = s[i]
            if (flags[0] and s[i]=='I') or (flags[1] and s[i]=='X') or (flags[2] and s[i]=='C') :
                res-= roman_dict[s[i]]
                i-=1
                flags = [False, False, False]  # I,X,C
            else:
                res+= roman_dict[s[i]]
                if s[i] == 'V' or s[i] == 'X':
                    flags = [True, False, False]
                elif s[i] == 'L' or s[i] == 'C':
                    flags = [False, True, False]
                elif s[i] == 'D' or s[i] == 'M':
                    flags = [False, False, True]
                i-=1
        return res

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ""
        if len(strs) == 1:
            return strs[0]
        #strs = sorted(strs,key = len)
        st = strs[0]
        max_st = ""
        for j in range(0,len(st)):
            curr_sub = st[0:j+1]
            in_strs = True
            for s in strs:
                try:
                    if curr_sub != s[0:j + 1]:
                        in_strs = False
                        break
                except:
                    break
            if in_strs == False: break
            max_st = curr_sub
        return max_st

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        if n<3:
            return []
        if n == 3 :
            if nums[0] + nums[1] + nums[2] == 0: return [nums]
            else: return []
        res = []
        for i in range(0,n):
            for j in range(0,n):
                if j!= i:
                    for k in range(0,n):
                        if k!= j and k!= i:
                            if nums[k]+nums[j]+nums[i] == 0:
                                flag = True
                                temp_lst = [nums[k],nums[j],nums[i]]
                                temp_lst.sort()
                                if temp_lst not in res: res.append(temp_lst)
        return res

    def threesum_2(self,nums):
        n = len(nums)
        if n < 3:
            return []
        if n == 3:
            if nums[0] + nums[1] + nums[2] == 0:
                return [nums]
            else:
                return []
        res = []
        arr = sorted(nums)
        i = 0
        j = 1
        k = n - 1
        while (i < k):
            j = i + 1
            k = n - 1
            while j < k:
                solu = [arr[i], arr[j], arr[k]]
                sum = solu[0] + solu[1] + solu[2]
                if sum == 0:
                    if solu not in res:
                        res.append(solu)
                    j += 1
                if sum < 0:
                    prev = arr[j]
                    try:
                        while sum - prev + arr[j] < 0:
                            j += 1
                    except:
                        break  # all out of range
                if sum > 0:
                    prev = arr[k]
                    try:
                        while sum - prev + arr[k] > 0:
                            k -= 1
                    except:
                        break  # all out of range
            i += 1
        return res

    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        n = len(nums)
        if n == 3:
            return nums[0]+nums[1]+nums[2]
        nums = sorted(nums)
        i = 0
        j = 1
        k = n-1
        close_sum = None
        while i<k:
            j = i+1
            while(j<k):
                sum = nums[i]+nums[j]+nums[k]
                if sum == target:
                    return sum
                elif close_sum ==None or abs(sum-target) < abs(close_sum-target):
                    close_sum = sum
                if sum >target:
                    k-=1
                elif sum<target:
                    j+=1
            k = n-1
            i+=1
        return close_sum

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        digits = digits.strip()
        if digits =="":
            return []
        strings = [['a','b','c'],['d','e','f'],['g','h','i'],['j','k','l'],['m','n','o'],['p','q','r','s'],['t','u','v'],['w','x','y','z']]
        res = []
        combos = []
        for i in range(len(digits)):
            combos.append(strings[int(digits[i])-2])
        for combo in combos:
            res = self.create_com(combo,res)
        return res

    def create_com(self,letters,res):
        if res ==[]:#list empty- create
            for lt in letters:
                res.append(lt)
            return res
        else:
            new_res = []
            for result in res:
                for lt in letters:
                    new_res.append(result+lt)
            return new_res

    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        n = len(nums)
        if n<4:
            return []
        elif n== 4:
            if nums[0]+nums[1]+nums[2]+nums[3]==target:
                return [nums]
            else: return []
        nums = sorted(nums)
        res = []
        a=0
        b=a+1
        d= n-1
        c= d-1
        while(a<n-3 ):
            b = a + 1
            d = n - 1
            c = d - 1
            while(b<c):
                curr_sum = nums[a] + nums[b] + nums[c] + nums[d]
                if curr_sum == target:
                    solu = [nums[a], nums[b], nums[c], nums[d]]
                    if solu not in res and a != b != c != d:
                        res.append(solu)
                    b += 1
                if curr_sum > target:  # need to decrease sum
                    if c == d - 1 and b == c - 1:  # c and d cant decrease - stop loop
                        break
                    elif b == c - 1:  # c cant decrease -> decrease d
                        d -= 1
                        c = d-1
                    # now if both can decrease or d cant decrease
                    else:
                        c-=1
                if curr_sum < target:  # need to decrease sum
                    if a == b - 1 and b == c - 1:  # c and d cant decrease - stop loop
                        break
                    elif b == c - 1:  # b cant increase -> increase a
                        a += 1
                        b = a+1
                    # now if both can increase:
                    else:
                        b+=1
            a+=1
        return res

    def removeNthFromEnd(self, head, n):
        """
        :type head: Optional[ListNode]
        :type n: int
        :rtype: Optional[ListNode]
        """
        if head is None: return None
        if head.next is None and n==1: return None
        father =self.recursive_try(head,n,False)
        if type(father)== int:
            return head.next
        son = father.next
        if son.next:
            father.next = son.next
            son.next = None
        else :
            father.next = None
        return head


    def recursive_try(self,node,n,flag):
        #function returns father of node to remove
        if node.next == None:
            flag = True
            return n-1
        else:
            left = self.recursive_try(node.next,n,False)
            if type(left)!= int:
                return left
            if left == 0:
                return node
            else : return left-1

    def removeNthFromEnd_2(self, head, n):
        """
        :type head: Optional[ListNode]
        :type n: int
        :rtype: Optional[ListNode]
        """
        node = head
        delete = head
        prev = None
        n-=1
        while(n!=0):
            node=node.next
            n-=1
        while(node.next!= None):
            node = node.next
            prev = delete
            delete = delete.next
        if not prev:#delete first node
            return head.next
        elif delete.next ==None:
            prev.next = None
        else:
            prev.next = delete.next
            delete.next=None
        return head

    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        lst = []
        i= 0
        while(i<len(s)):
            char = s[i]
            if char =="[" or char == "{" or char =="(":
                lst.append(char)
            else:
                if len(lst)== 0:
                    return False
                check_char = lst.pop()
                if ord(s[i]) -ord(check_char) <1 or ord(s[i]) -ord(check_char) >2  :
                    return False
            i+=1
        if len(lst)==0 : return True
        else: return False

    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """



    # need to update - q


#need to update - q 18 4Sum
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

sol = Solution()
e = ListNode(5)
d = ListNode(4,e)
c = ListNode(3,d)
b = ListNode(2,c)
h = ListNode(1,b)

#h = sol.removeNthFromEnd_2(h,2)
#while h :
 #   print(h.val,",")
  #  h=h.next

print(sol.isValid("({([])})"))











