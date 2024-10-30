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
        if not list2 :
            return list1
        if not list1:
            return list2
        #both lists exsists
        n1 = list1
        n2 = list2
        if n1.val>n2.val:
            head = ListNode(n2.val)
            n2 = n2.next
        else:
            head = ListNode(n1.val)
            n1 = n1.next
        node = head
        while n1 and n2:
            if n1.val < n2.val:
                node.next = ListNode(n1.val)
                n1=n1.next
            else:
                node.next = ListNode(n2.val)
                n2 = n2.next
            node = node.next
        if n1:
            node.next = n1
        if n2:
            node.next = n2
        return head

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        j = 1
        for i in range(len(nums) - 1):
            if nums[i] != nums[i + 1]:
                nums[j] = nums[i + 1]
                j += 1
        return j, nums



        # need to update - q

    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        while(n!=0):
            if res == []:
                res.append("()")
            else:
                new_res = []
                for st in res:
                    new_res.append("("+st+")")
                    if (st+"()" ) not in new_res:
                        new_res.append(st+"()" )
                    if ("()"+st) not in new_res:
                        new_res.append("()"+st)
                    res = new_res
            n-=1
        return res

    def mergeKLists(self, lists):
        """
        :type lists: List[Optional[ListNode]]
        :rtype: Optional[ListNode]
        """
        n=len(lists)
        i=0
        while i< n:
            if not lists[i]:
                lists.pop(i)
                n-=1
            else : i+=1
        try:
            lists = sorted(lists, key=lambda lst: lst.val if lst else lists.remove(lst))  # sort by first val
            head = ListNode(lists[0].val)
            if lists[0].next == None:
                lists.pop(0)
            else:
                lists[0] = lists[0].next
            node = head
        except:
            return None
        while(lists):
            lists = sorted(lists, key=lambda lst: lst.val)  # sort by first val
            node.next = ListNode(lists[0].val)
            node = node.next
            if lists[0].next == None:
                lists.pop(0)
            else: lists[0] = lists[0].next
        return head


    def mergeKLists_2(self, lists):
        """
        :type lists: List[Optional[ListNode]]
        :rtype: Optional[ListNode]
        """
        data_val = []
        data = {}
        for head in lists:
            temp = head
            while temp:
                val = temp.val
                if val not in data:
                    data[val] = []
                    data_val.append(val)
                data[val].append(temp)
                temp = temp.next
        data_val.sort()
        head = None
        temp = head
        for value in data_val:
            node_lst = data[value]
            for node in node_lst:
                if not head:
                    head =node
                    temp = node
                else:
                    temp.next = node
                    temp = temp.next
        return head

    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        i = 0
        n = len(nums)
        print(nums)
        while (i<n):
            if(nums[i]==val):
                if i == n-1:
                    return n-1
                nums[i:n-1] = nums[i+1:n]
                nums[n-1] = val
                n-=1
            else: i+=1
        return n

    def removeElement_2(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        id1 = len(nums)-1
        for id2 in range(id1,-1,-1):
            if nums[id2]==val:
                nums[id2] = nums[id1]
                nums[id1]= val
                id1-=1
        return id1+1

    def removeElement_3(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        k =0
        for i in range(len(nums)):
            if nums[i]!= val:
                nums[k] = nums[i]
                k+=1
        return k

    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        for i in range(len(haystack)):
            if haystack[i] == needle[0]:
                flag = True
                for k in range(1,len(needle)):
                    try:
                        if haystack[i + k] != needle[k]:
                            flag = False
                    except:
                        flag = False
                        break
                if flag : return i
        return -1

    def strStr_2(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle in haystack:
            for i in range (0,len(haystack)):
                if haystack[i:(i+len(needle))]== needle:
                    return i
        else : return -1

    def swapPairs(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        if not head or not head.next:
            return head
        else:
            node1 = head
            node2 = head.next
            prev = None
            while node1 is not None and node2 is not None:
                if not prev:
                    node1.next = node2.next
                    head = node2
                    node2.next = node1
                else:
                    node1.next = node2.next
                    prev.next = node2
                    node2.next = node1
                prev = node1
                node1 = node1.next
                if not node1 : break
                node2 =node1.next
        return head

    def reverseKGroup(self, head, k):
        """
        :type head: Optional[ListNode]
        :type k: int
        :rtype: Optional[ListNode]
        """
        if k == 1 or head == None or head.next == None:
            return head
        end = head
        while(True):
            start = end
            lst = []
            try:
                for i in range(k):
                    lst.append(end)
                    end = end.next
                new_start = lst.pop()
                node = new_start
                node.next = None
                while (lst):
                    node.next = lst.pop()
                    node = node.next
                    node.next = None
                node.next = end
                if start == head:
                    head = new_start
                    prev = node
                else :
                    prev.next = new_start
                    prev = node
            except :return head
        return head

    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if (divisor>0 and dividend >0) or (divisor<0 and dividend<0) : sign =1
        else: sign =-1
        div_abs = abs(divisor)
        divid_abs = abs(dividend)
        if divid_abs > ((2**31) -1) and sign==1 :
            return (2**31) -1 *sign
        if divid_abs > (2**31) and sign ==-1 :
            return (2**31) *sign
        count = 0 *sign
        if div_abs ==1:
            return divid_abs*sign
        sum = 0
        while(sum<=divid_abs):
            sum+=div_abs
            count+=1
        return (count-1)*sign

    def divide_2(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if (divisor >0 and dividend >=0) or (dividend<=0 and divisor<0):
            sign =1
        else: sign =-1
        dividend = abs(dividend)
        divisor = abs(divisor)
        res = len(range(0,dividend-divisor+1,divisor))
        if sign==-1:
            res = -res
        minus_lim = -(2**31)
        plus_lim = (2**31-1)
        return min(max(res,minus_lim),plus_lim)

    def divide_3(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if (divisor >0 and dividend >=0) or (dividend<=0 and divisor<0):
            sign =1
        else: sign =-1
        dividend = abs(dividend)
        divisor = abs(divisor)
        res = 0
        while(dividend>=divisor):
            n=1
            add = divisor
            while(dividend>=add):
                res+=n
                dividend-=add
                add+=add
                n+=n
        if sign<0:
            res = -res
        minus_lim = -(2 ** 31)
        plus_lim = (2 ** 31 - 1)
        return min(max(res, minus_lim), plus_lim)

    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        s_id = 0
        s_check = 0
        s_len = len(s)
        words_len = len(words[0])
        words_count = len(words)
        if s_len < len(words):
            return []
        res=[]
        words_set = set(words)
        if len(words_set)==1:#all same word
            pass
        for i in range(s_len):
            for word in words:
                s_check = i
                flag = self.checksubin(s,word,i)
                if flag:
                    remain_words = list(words)
                    remain_words.remove(word)
                    s_check +=words_len
                    k=0
                    while len(remain_words)!=0 and k< (len(remain_words)):
                        flag = self.checksubin(s, remain_words[k], s_check)
                        if flag:
                            s_check+=words_len
                            remain_words.pop(k)
                            k=0
                        else: k+=1
                    if remain_words==[]:
                        if i not in res: res.append(i)
        return res

    def checkword(self,s,word,amount):
        index = 0
        checked_index = index
        s_ln = len(s)
        word_ln = len(word)
        amount*= word_ln
        res =[]
        count = 0
        while(checked_index< s_ln):
            checked_index = index
            for k in range(word_ln):
                if s[checked_index]== word[k]:
                    checked_index+=1
                    count = checked_index
                else:
                    count = 0
                    break
            if count == amount:
                count = 0
                res.append(index)
            index+=1
        return res

    def checksubin(self,s,word,id):
        try:
            for k in range(len(word)):
                if s[k + id] != word[k]:
                    return False
            return True
        except: return False

    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if target<= nums[0]:
            return 0
        n = len(nums)
        if target == nums[n-1]: return n-1
        if target > nums[n-1]:
            return n
        return self.recursivesol(nums,target,0,n-1)

    def recursivesol(self,nums,target,start,end):
        if start == end:
            if nums[start]== target:
                return start
            elif nums[start]>target:
                return start# will bee in this position after
            else:
                return end+1
        else:
            if end-1 == start:# 2 left
                if target<= nums[start]:
                    return start
                elif target <= nums[end]:
                    return end
                else : return end+1
            mid = (start+end)//2
            if nums[mid]== target:
                return mid
            elif nums[mid]<target:
                return self.recursivesol(nums,target,mid,end)
            else:
                return self.recursivesol(nums,target,start,mid-1)

    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        words = s.split()
        return len(s.split()[-1])

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) ==1 :
            return True
        id = 0
        while(id<len(nums)):
            jumps = nums[id]
            if (jumps+id >=len(nums)-1):
                return True
            max = [-1, -1]  # id ,jums on id
            for k in range(1, jumps + 1):
                try:
                    if max[1] <= (nums[k + id] + id):
                        max = [k + id, nums[k + id]]
                except: break
            if max[1]<=0: return False
            id += max[1]-1
        if id >=len(nums): return True
        return False

    def canJump_2(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums)== 0:
            return True
        else:
            return self.recursive_jump(nums,0)

    def recursive_jump(self,nums,id):
        n = len(nums)
        if id >= n-1: return True #means id reached is the last one or more
        else:
            jumps = nums[id]
            if jumps == 0 :
                return False
            max_id = id+jumps
            while(max_id>id):
                if self.recursive_jump(nums,max_id): return True
                max_id-=1
                curr_jump= id + jumps
                while (id + jumps)>=(nums[max_id] + max_id)and max_id>id:#to skip unneded parts
                    max_id-=1
        return False







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
#h = sol.mergeTwoLists(e,c)
#h = sol.removeNthFromEnd_2(h,2)
#h = sol.mergeKLists_2([None,e])

h = sol.reverseKGroup(h,2)
#while h:
 #   print(h.val, ",")
  #  h = h.next
#a = [0,0,1,1,1,2,2,3,3,4]
#len,a  =sol.removeDuplicates(a)
#print(a)

print(sol.canJump_2([5,9,3,2,1,0,2,3,3,1,0,0]))













