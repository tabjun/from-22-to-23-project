-- ------------------------------
-- 1절
-- ------------------------------

use market_db;
use sys;

use market_db;
# member 테이블이 존재하는 db를 불러와야 함.
# use sys를 실행하고 member 테이블을 불러오면 에러가 발생함(경로 설정이랑 같음)
select * 
from member 
where mem_name = '블랙핑크' ; # 세미콜론 전까지는 한 문장

select * 
from market_db.member # 현재 사용 중인 db말고 다른데서 할 때는 앞에 db이름 설정해줘야함
where mem_name = '블랙핑크' ; # 세미콜론 전까지는 한 문장

# 보고싶은 열만 부를 때
select addr, height, debut_date # 꼭 순서대로 안 불러와도 됨 
from member 
where mem_name = '블랙핑크' ; # 세미콜론 전까지는 한 문장

SELECT mem_name FROM member;

SELECT addr, debut_date, mem_name
	FROM member;

# 변수명 옆에 별칭을 붙여줄 수 있음(파이썬의 as같은 것)
# 말 그대로 별칭, 실행할 때만 보이는 거임. 변수명이 변경되는 것은 아님
SELECT addr 주소, debut_date "데뷔 일자", mem_name # 띄어쓰기가 있는 경우 따옴표로 묶어주면 됨
	FROM member;

SELECT * FROM member WHERE mem_name = '블랙핑크';

SELECT * FROM member WHERE mem_number = 4;

# 조건에 쓰는 변수가 select문을 사용할 때 없어도 됨
SELECT mem_id, mem_name 
	FROM member 
	WHERE height <= 162;

SELECT mem_name, height, mem_number 
	FROM member 
	WHERE height >= 165 AND mem_number > 6;

SELECT mem_name, height, mem_number 
	FROM member 
	WHERE height >= 165 OR mem_number > 6;

SELECT mem_name, height 
	FROM member 
	WHERE height >= 163 AND height <= 165;

SELECT mem_name, height 
   FROM member 
   WHERE height BETWEEN 163 AND 165;
   
SELECT mem_name, addr 
   FROM member 
   WHERE addr = '경기' OR addr = '전남' OR addr = '경남';
   
# or 대신 in을 사용하면 더 간단하게 표현 가능
SELECT mem_name, addr 
   FROM member 
   WHERE addr IN('경기', '전남', '경남');
   
# 문자형 조건 조회할 때 쓰는 함수 like 
SELECT * 
   FROM member 
   WHERE mem_name LIKE '우%'; # 우로 시작하는 이름


SELECT * 
   FROM member 
   WHERE mem_name LIKE '__핑크'; # 언더바(_)는 한 글자를 의미, 여기서는 00핑크
   
SELECT height FROM member WHERE mem_name = '에이핑크';

SELECT mem_name, height FROM member WHERE height > 164;

SELECT mem_name, height FROM member
	WHERE height > (SELECT height FROM member WHERE mem_name = '에이핑크');

----------------------------------------------------------
# 정렬 함수 order by
# select 열 이름 from 테이블 이름 where 조건식 group by 열 이름 
# having 조건식 order by 열 이름 limit 개수;
---------------------------------------------------------- 
# 데뷔일자 순서대로
# 기본적으로 오름차순
SELECT mem_id, mem_name, debut_date 
   FROM member 
   ORDER BY debut_date;

# 내림차순 order by 변수명 desc;
SELECT mem_id, mem_name, debut_date 
   FROM member 
   ORDER BY debut_date DESC;
   
SELECT mem_id, mem_name, debut_date, height
   FROM member 
   ORDER BY height DESC
   WHERE height  >= 164;  -- 오류 발생, 순서가 잘못 됨 where가 먼저임 where로 결과 조회한 다음 그 결과 속에서 정렬
   
SELECT mem_id, mem_name, debut_date, height
   FROM member 
   WHERE height  >= 164
   ORDER BY height DESC;
   
# 섞어 쓰는 것도 가능
SELECT mem_id, mem_name, debut_date, height
   FROM member 
   WHERE height  >= 164
   ORDER BY height DESC, debut_date ASC; # order by 첫 변수 먼저 정렬 뒤, 동률 일 때 뒤에 변수로 정렬 따라감

   
# 상위 3개 행만 보여줘 head()와 같은 기능
SELECT *
   FROM member 
   LIMIT 3;
   
SELECT mem_name, debut_date
   FROM member
   ORDER BY debut_date
   LIMIT 3;
   
# 3번째부터 2개 행만 보여줘 => 4,5
SELECT mem_name, height
   FROM member
   ORDER BY height DESC
   LIMIT 3,2;
      
SELECT addr FROM member;

SELECT addr FROM member ORDER BY addr;

# 중복된 값 제외하고 한개씩만 보여줌
SELECT DISTINCT addr FROM member;

-- ------------------------------
-- 2절
# group by 
# select 열 이름 from 테이블 이름 where 조건식 group by 열 이름 having 조건식 order by 열 이름 limit 개수;
-- ------------------------------

SELECT mem_id, amount FROM buy ORDER BY mem_id;

SELECT mem_id, SUM(amount) FROM buy GROUP BY mem_id;

# 별칭줘서 깔끔하게, 실행 시 조회 테이블에만 적용됨
SELECT mem_id "회원 아이디", SUM(amount) "총 구매 개수"
   FROM buy GROUP BY mem_id;

# 집계 함수 내에서 변수끼리 연산 가능
select mem_id "회원 아이디", sum(price*amount) "총 구매 금액"
   from buy group by mem_id;
   
select avg(amount) "평균 구매 개수" from buy;

SELECT mem_id, AVG(amount) "평균 구매 개수" 
	FROM buy
	GROUP BY mem_id;
    
SELECT COUNT(*) FROM member;

# 결측 제외
SELECT COUNT(phone1) "연락처가 있는 회원" FROM member;

select mem_id "회원 아이디", sum(price*amount) "총 구매 금액"
   from buy
   group by mem_id;

# where 절에서는 집계 함수 불가
# goroup by 뒤에 having 절 사용
SELECT mem_id "회원 아이디", SUM(price*amount) "총 구매 금액"
   FROM buy 
   WHERE SUM(price*amount) > 1000 
   GROUP BY mem_id; -- 에러 발생함

SELECT mem_id "회원 아이디", SUM(price*amount) "총 구매 금액"
   FROM buy 
   GROUP BY mem_id   
   HAVING SUM(price*amount) > 1000 ;

SELECT mem_id "회원 아이디", SUM(price*amount) "총 구매 금액"
   FROM buy 
   GROUP BY mem_id   
   HAVING SUM(price*amount) > 1000
   ORDER BY SUM(price*amount) DESC;

-- ------------------------------
-- 3절
-- ------------------------------

USE market_db;
CREATE TABLE hongong1 (toy_id  INT, toy_name CHAR(4), age INT);
INSERT INTO hongong1 VALUES (1, '우디', 25); # 각 변수 자리에 맞게 입력

# not null으로 설정하지 않았다면 굳이 3개 다 넣지 않아도 됨
INSERT INTO hongong1(toy_id, toy_name) VALUES (2, '버즈');

INSERT INTO hongong1(toy_name,age, toy_id) VALUES ('제시', 20, 3);

# 자동으로 값 넣고 싶다 auto_increment 이 때 꼭 pk로 설정해야함
CREATE TABLE hongong2 ( 
   toy_id  INT AUTO_INCREMENT PRIMARY KEY, 
   toy_name CHAR(4), 
   age INT);

# NULL은 자동으로 값이 들어감
# 물건 구매할 때, 물건을 구매한 순서대로 이력에 담기지만 내가 한적이 없음, 그게 다 사용자가 직접 입력안해도
# 자동으로 값이 매겨지는 auto_increment
INSERT INTO hongong2 VALUES (NULL, '보핍', 25);
INSERT INTO hongong2 VALUES (NULL, '슬링키', 22);
INSERT INTO hongong2 VALUES (NULL, '렉스', 21);
SELECT * FROM hongong2;

# auto_increment로 값입력된게 어디까지 됐는데 확인해볼때
SELECT LAST_INSERT_ID(); 

# 내가 원하는 값으로 시작하고 싶을 때
ALTER TABLE hongong2 AUTO_INCREMENT=100;
INSERT INTO hongong2 VALUES (NULL, '재남', 35);
SELECT * FROM hongong2;

CREATE TABLE hongong3 ( 
   toy_id  INT AUTO_INCREMENT PRIMARY KEY, 
   toy_name CHAR(4), 
   age INT);
ALTER TABLE hongong3 AUTO_INCREMENT=1000;

# 값 입력될 때마다 3씩 증가
SET @@auto_increment_increment=3;

INSERT INTO hongong3 VALUES (NULL, '토마스', 20);
INSERT INTO hongong3 VALUES (NULL, '제임스', 23);
INSERT INTO hongong3 VALUES (NULL, '고든', 25);
SELECT * FROM hongong3;

# 다른 db에서 테이블 가져오기
SELECT COUNT(*) FROM world.city;

# DESC 그냥 테이블 정보 조회할 때(파이썬 info()와 비슷)
# 그냥 desc로만 쓰면 아마 describe()인듯
DESC world.city;

SELECT * FROM world.city LIMIT 5;

# 테이블 삭제
DROP table if EXISTS city_popul;
# 우리가 쓰는 db로 테이블 가져오기(지금은 market_db)
CREATE TABLE city_popul ( city_name CHAR(35), population INT);

# insert로 가져오는데, world.city 테이블에서 name과 population만 가져오기
INSERT INTO city_popul
    SELECT Name, Population FROM world.city;

DESC city_popul;
SELECT COUNT(*) FROM city_popul;

select * from city_popul limit 5;

USE market_db;

select * from city_popul where city_name = 'Seoul';
UPDATE city_popul
    SET city_name = '서울'
    WHERE city_name = 'Seoul';
SELECT  * FROM city_popul WHERE  city_name = '서울';

select * from city_popul where city_name = 'New York';
UPDATE city_popul
    SET city_name = '뉴욕', population = 0 # 이름 한글, 인구 0으로 바꾸기
    WHERE city_name = 'New York'; # 만약 where이 없으면 모든 행이 바뀜
SELECT  * FROM city_popul WHERE  city_name = '뉴욕';

-- UPDATE city_popul
--    SET city_name = '서울'

# 만명 단위로 끊고 싶을 때
# 밑의 코드는 where이 없으니까 전체 행에 적용
# 만약 vscode에서 실행하면 where 없다고 경고문 나옴
# 그만큼 조심해서 코딩해야 함
UPDATE city_popul
    SET population = population / 10000 ;
SELECT * FROM city_popul LIMIT 5;

# 조건에 해당하는 모든 행 삭제
DELETE FROM city_popul 
    WHERE city_name LIKE 'New%';

# 조건에 해당하는 행 5개만 삭제
DELETE FROM city_popul 
    WHERE city_name LIKE 'New%'
    LIMIT 5;

# 각 다른 db에서 불러온 테이블을 하나의 테이블로 합쳐줄 수 있음
# create table 테이블 이름 (select * from db이름.A테이블 이름, db이름.B테이블 이름);
# 변수명은 A 테이블의 변수부터 
CREATE TABLE big_table1 (SELECT * FROM world.city , sakila.country);
select * from world.city limit 5;
select * from sakila.country limit 5;
select * from big_table1 limit 5;
CREATE TABLE big_table2 (SELECT * FROM world.city , sakila.country); 
CREATE TABLE big_table3 (SELECT * FROM world.city , sakila.country); 
SELECT COUNT(*) FROM big_table1;

# 데이터 삭제
# delete, drop, truncate
# delete와 truncate는 테이블이 남아있고 drop은 테이블 자체가 삭제됨
# delete와 truncate의 차이점은 delete는 where절을 이용해서 특정 행만 삭제 가능
# truncate는 테이블 전체 행 삭제
# delete는 테이블은 남아있고 행만 삭제
DELETE FROM big_table1;
select * from big_table1;
# drop은 테이블 자체를 삭제
DROP TABLE big_table2;
# truncate는 테이블은 남아있고 행만 삭제
TRUNCATE TABLE big_table3;

