-- ------------------------------
-- 1절
-- ------------------------------

USE market_db;
CREATE TABLE hongong4 (
    tinyint_col  TINYINT, # - 128 ~ 127
    smallint_col SMALLINT, # - 32768 ~ 32767
    int_col    INT, # - 2147483648 ~ 2147483647
    bigint_col BIGINT ) # - 9000000000000000000 ~ 9000000000000000000; 

select * from hongong4;
desc hongong4;
INSERT INTO hongong4 VALUES(127, 32767, 2147483647, 9000000000000000000);

# 128은 tinyint의 범위를 벗어나서 에러가 발생
INSERT INTO hongong4 VALUES(128, 32768, 2147483648, 90000000000000000000);

DROP TABLE IF EXISTS buy, member;
CREATE TABLE member -- 회원 테이블
( mem_id      CHAR(8) NOT NULL PRIMARY KEY, -- 회원 아이디(PK)
  mem_name        VARCHAR(10) NOT NULL, -- 이름
  mem_number    INT NOT NULL,  -- 인원수
  addr          CHAR(2) NOT NULL, -- 주소(경기,서울,경남 식으로 2글자만입력)
  phone1        CHAR(3), -- 연락처의 국번(02, 031, 055 등)
  phone2        CHAR(8), -- 연락처의 나머지 전화번호(하이픈제외)
  height        SMALLINT,  -- 평균 키
  debut_date    DATE  -- 데뷔 일자
);

# int는 21억까지 입력가능한데, 인원수는 그렇게 많이 필요없으니 tinyint로 설정
# 그렇다고 int가 틀린 건 아님 굳이 4바이트를 사용할 필요가 없음
# tinyint는 2바이트, 2^8 = 256까지 입력가능
# int는 4바이트, 2^32 = 4294967296까지 입력가능
# smallint는 2바이트, 2^16 = 65536까지 입력가능
# 키는 tinyint로 설정하면 최대값이 128로 200이 넘어가면 에러가 발생, 이 때 usigned를 사용하면 0 ~ 255까지 입력가능
DROP TABLE IF EXISTS member;
CREATE TABLE member -- 회원 테이블
( mem_id      CHAR(8) NOT NULL PRIMARY KEY, -- 회원 아이디(PK)
  mem_name        VARCHAR(10) NOT NULL, -- 이름
  mem_number    TINYINT  NOT NULL,  -- 인원수
  addr          CHAR(2) NOT NULL, -- 주소(경기,서울,경남 식으로 2글자만입력)
  phone1        CHAR(3), -- 연락처의 국번(02, 031, 055 등)
  phone2        CHAR(8), -- 연락처의 나머지 전화번호(하이픈제외)
  height        TINYINT UNSIGNED,  -- 평균 키
  debut_date    DATE  -- 데뷔 일자
);

DROP TABLE IF EXISTS member;
CREATE TABLE member -- 회원 테이블
( mem_id      CHAR(8) NOT NULL PRIMARY KEY, -- 회원 아이디(PK)
  mem_name        VARCHAR(10) NOT NULL, -- 이름
  mem_number    TINYINT  NOT NULL,  -- 인원수
  addr          CHAR(2) NOT NULL, -- 주소(경기,서울,경남 식으로 2글자만입력)
  phone1        CHAR(3), -- 연락처의 국번(02, 031, 055 등)
  phone2        CHAR(8), -- 연락처의 나머지 전화번호(하이픈제외)
  height        TINYINT UNSIGNED,  -- 평균 키
  debut_date    DATE  -- 데뷔 일자
);

# char는 고정길이 문자열, varchar는 가변길이 문자열
# char는 최대 255글자까지 입력가능, varchar는 최대 16383글자 입력가능
# 만약 char(10)을 설정하고 3글자를 입력하면 나머지 7글자는 공백으로 채워짐
# 반면 varchar(10)을 설정하면 3글자만 입력되고 나머지는 공백으로 채워지지 않음: 3글자만큼만 저장공간을 차지 
CREATE TABLE big_table (
  data1  CHAR(255),
  data2  VARCHAR(16384) ); # 에러 발생(최대 16383글자까지 입력가능)

# 이거 두개 한번에 하면 에러 발생
# 따로 하면 괜찮음
CREATE TABLE big_table (
  data1  CHAR(255),
  data2  VARCHAR(16383) );


CREATE TABLE big_table (
  data1  CHAR(255)
);
drop table big_table;

CREATE TABLE big_table (
  data2  VARCHAR(16383) );
drop table big_table;

CREATE DATABASE netflix_db;
USE netflix_db;
use netflix_db;
CREATE TABLE movie 
  (movie_id        INT,
   movie_title     VARCHAR(30),
   movie_director  VARCHAR(20),
   movie_star      VARCHAR(20),
   movie_script    LONGTEXT, # 42억 9천만까지 입력가능: 대본의 자막 등 입력할 때 사용
   movie_film      LONGBLOB # binarylargeobject: 영화 파일 등 입력할 때 사용
); 
drop table movie;

# 날짜형
# date: 날짜만, yyyymmdd
# time: 시간만, hhmmss
# datetime: 날짜와 시간, yyyymmddhhmmss


# 변수 선언 및 초기화
# 변수 선언은 휘발성 변수로 선언하면 선언한 쿼리가 끝나면 사라짐
# set @변수명 = 값; 변수 선언 및 값 대입
# select @변수명; 변수 값 출력
USE market_db;
SET @myVar1 = 5 ;
SET @myVar2 = 4.25 ;

# 종료했다가 다시 실행하면 변수가 초기화됨
# 변수는 출력되는데 값은 null로 출력
SELECT @myVar1 ;
SELECT @myVar1 + @myVar2 ;

# 파이썬 f포맷팅 출력이랑 비슷
SET @txt = '가수 이름==> ' ;
SET @height = 166;
SELECT @txt , mem_name # 선언해둔 값 '가수 이름==> ' 출력, 그리고 member 테이블에서 조건에 맞는 멤버 이름 출력
FROM member WHERE height > @height ; # 미리 선언해둔 @키(166)보다 큰 키를 가진 회원의 이름 출력

set @cco = '에베베:'; set @keke = '뾱뾱 :';
select @cco, mem_name, @keke, height from member where addr = '서울';

# limit 사용할 때 선언한 변수 사용하면 에러 발생
set @count = 2;
select @cco, mem_name, @keke, height 
from member where addr = '서울' limit @count; 

SET @count = 3;
SELECT mem_name, height FROM member ORDER BY height LIMIT @count;

# prepare: 쿼리문을 미리 준비해두고 나중에 실행할 때 사용
# execute: 준비해둔 쿼리문을 실행
SET @count = 3;
# prepare 쿼리문을 미리 준비해두고 limit 뒤에 들어오는 함수는 모른다 '?'를 사용
PREPARE mySQL FROM 'SELECT mem_name, height FROM member ORDER BY height LIMIT ?';
# execute 준비해둔 쿼리문을 실행 이 때 지정해둔 ?에 대한 값을 using으로 대입
EXECUTE mySQL USING @count;

set @qqq = '*';
prepare myqqq from 'select ?, height from member order by height';
execute myqqq using @qqq; # using쓸 때는 미리 선언해둔 변수를 사용해야 함

# 데이터 형식 변환
# cast(값 as 데이터형식[길이])
# convert(값, 데이터형식[길이])
SELECT AVG(price) '평균 가격' FROM buy;

# signed: 부호가 있는 정수
# 이 문장에서는 반올림 같은 개념
SELECT CAST(AVG(price) AS SIGNED)  '평균 가격'  FROM buy ;
-- 또는
SELECT CONVERT(AVG(price) , SIGNED)  '평균 가격'  FROM buy ;

SELECT CAST('2022$02$02' AS DATE);
SELECT CAST('2022/02/02' AS DATE);
SELECT CAST('2022%02%02' AS DATE);
SELECT CAST('2022@02@02' AS DATE);

select cast('2022-01-01' as date);
select convert('2201.11.12', date);

# concat은 문자열 연결할 때, cast로 price를 문자형으로 변경해서 값 출력
SELECT num, CONCAT(CAST(price AS CHAR), 'X', CAST(amount AS CHAR) ,'=' ) '가격X수량',
    price*amount '구매액' 
  FROM buy ;

SELECT '100' + '200' ; -- 문자와 문자를 더함 (정수로 변환되서 연산됨)
SELECT CONCAT('100', '200'); -- 문자와 문자를 연결 (문자로 처리)
SELECT CONCAT(100, '200'); -- 정수와 문자를 연결 (정수가 문자로 변환되서 처리)
SELECT 1 > '2mega'; -- 정수인 2로 변환되어서 비교
SELECT 3 > '2MEGA'; -- 정수인 2로 변환되어서 비교
SELECT 0 = 'mega2'; -- 문자는 0으로 변환됨

-- ------------------------------
-- 2절
-- ------------------------------

# 조인
drop database market_db;
use market_db;

SELECT * 
   FROM buy
     INNER JOIN member
     ON buy.mem_id = member.mem_id
   WHERE buy.mem_id = 'GRL';

SELECT * 
   FROM buy
     INNER JOIN member
     ON buy.mem_id = member.mem_id;
     
# 오류가 발생함. select문에서 mem_id가 두 테이블에 모두 존재하기 때문에 어느 테이블의 mem_id인지 모름
SELECT mem_id, mem_name, prod_name, addr, CONCAT(phone1, phone2) AS '연락처' 
   FROM buy
     INNER JOIN member
     ON buy.mem_id = member.mem_id;

# 고친 코드
select phone1, phone2 from member;
select buy.mem_id, mem_name, prod_name, addr, concat(phone1, phone2) as '연락처'
from buy
inner join member
on buy.mem_id = member.mem_id;
# 지금은 inner join으로 구매한 이력이 있는 회원만 출력

# b. 처럼 테이블 명 앞에 별칭 붙이면, 나중에 테이블 명을 쓰지 않고 별칭만으로도 사용 가능
# from에서 buy테이블을 b로 별칭, join에서 member테이블을 m으로 별칭, 그리고 앞의 코드에 별칭 적용
SELECT B.mem_id, M.mem_name, B.prod_name, M.addr, 
        CONCAT(M.phone1, M.phone2) AS '연락처' 
   FROM buy B
     INNER JOIN member M
     ON B.mem_id = M.mem_id;
          
select m.mem_id, m.mem_name, b.prod_name, m.addr
from buy b inner join member m on b.mem_id = m.mem_id;

SELECT M.mem_id, M.mem_name, B.prod_name, M.addr
   FROM buy B
     INNER JOIN member M
     ON B.mem_id = M.mem_id
   ORDER BY M.mem_id;

select m.mem_name, b.prod_name, m.addr, concat(m.phone1, m.mem_name) as 'etc'
from buy b inner join member m on b.mem_id = m.mem_id;

# 전체 열에 대해 고유한 값만 출력
# a,b열 이름이 같더라도 c열의 값이 다르면 다른 행으로 인식
select distinct m.mem_name, b.prod_name, m.addr 
from buy b 
inner join member m 
on b.mem_id = m.mem_id
group by m.mem_name, b.prod_name, m.addr
order by m.addr;


SELECT DISTINCT M.mem_id, M.mem_name, M.addr
   FROM buy B
     INNER JOIN member M
     ON B.mem_id = M.mem_id
   ORDER BY M.mem_id;
   
SELECT M.mem_id, M.mem_name, B.prod_name, M.addr
   FROM member M
     LEFT OUTER JOIN buy B
     ON M.mem_id = B.mem_id
   ORDER BY M.mem_id;

SELECT M.mem_id, M.mem_name, B.prod_name, M.addr
   FROM buy B
     RIGHT OUTER JOIN member M
     ON M.mem_id = B.mem_id
   ORDER BY M.mem_id;
   
select distinct m.mem_id, m.mem_name, b.prod_name, m.addr
   from buy b
      right outer join member m
      on m.mem_id = b.mem_id
   order by m.mem_id;

SELECT DISTINCT M.mem_id, B.prod_name, M.mem_name, M.addr
   FROM member M
     LEFT OUTER JOIN buy B
     ON M.mem_id = B.mem_id
   WHERE B.prod_name IS NULL
   ORDER BY M.mem_id;

SELECT * 
   FROM buy 
     CROSS JOIN member ;

select count(*)
   from buy b
      right outer join member m
      on m.mem_id = b.mem_id
   order by m.mem_id;

SELECT COUNT(*) "데이터 개수"
   FROM sakila.inventory
      CROSS JOIN world.city;

SELECT COUNT(*) "데이터 개수"
   FROM sakila.inventory
      CROSS JOIN world.city;

CREATE TABLE cross_table
    SELECT *
       FROM sakila.actor
          CROSS JOIN world.country;

SELECT * FROM cross_table LIMIT 5;

USE market_db;

# 자체 조인인
CREATE TABLE emp_table (emp CHAR(4), manager CHAR(4), phone VARCHAR(8));

INSERT INTO emp_table VALUES('대표', NULL, '0000');
INSERT INTO emp_table VALUES('영업이사', '대표', '1111');
INSERT INTO emp_table VALUES('관리이사', '대표', '2222');
INSERT INTO emp_table VALUES('정보이사', '대표', '3333');
INSERT INTO emp_table VALUES('영업과장', '영업이사', '1111-1');
INSERT INTO emp_table VALUES('경리부장', '관리이사', '2222-1');
INSERT INTO emp_table VALUES('인사부장', '관리이사', '2222-2');
INSERT INTO emp_table VALUES('개발팀장', '정보이사', '3333-1');
INSERT INTO emp_table VALUES('개발주임', '정보이사', '3333-1-1');

# 자체 조인을 위해서, 하나의 테이블을 별칭을 다르게 해서 조인: 마치 테이블 2개를 사용하는 것처럼
SELECT A.emp "직원" , B.emp "직속상관", B.phone "직속상관연락처"
   FROM emp_table A
      INNER JOIN emp_table B
         ON A.manager = B.emp
   WHERE A.emp = '경리부장';

-- ------------------------------
-- 3절
-- ------------------------------

use market_db;
DROP PROCEDURE IF EXISTS ifProc1; -- 기존에 만든적이 있다면 삭제
# 만약 delimiter를 정의하지 않고 함수를 만들었다면, 세미콜론에 의해 개별 함수로 인식돼어 db에 저장됨
# delimiter를 정의해서 세미콜론이 있더라도 문장이 끝나지 않고 전체가 하나의 함수로 인식될 수 있음
DELIMITER $$
CREATE PROCEDURE ifProc1()
BEGIN
   IF 100 = 100 THEN  
      SELECT '100은 100과 같습니다.';
   END IF;
END $$
DELIMITER ;
CALL ifProc1();

DROP PROCEDURE IF EXISTS ifProc2; 
DELIMITER $$
CREATE PROCEDURE ifProc2()
BEGIN
   DECLARE myNum INT;  -- myNum 변수선언
   SET myNum = 200;  -- 변수에 값 대입
   IF myNum = 100 THEN  
      SELECT '100입니다.';
   ELSE
      SELECT '100이 아닙니다.';
   END IF;
END $$
DELIMITER ;
CALL ifProc2();


DROP PROCEDURE IF EXISTS ifProc3; 
DELIMITER $$
CREATE PROCEDURE ifProc3()
BEGIN
    DECLARE debutDate DATE; -- 데뷰일
    DECLARE curDate DATE; -- 오늘
    DECLARE days INT; -- 활동한 일수

    SELECT debut_date INTO debutDate -- debut_date 결과를 hireDATE에 대입
       FROM market_db.member
       WHERE mem_id = 'APN';

    SET curDATE = CURRENT_DATE(); -- 현재 날짜
    SET days =  DATEDIFF(curDATE, debutDate); -- 날짜의 차이, 일 단위

    IF (days/365) >= 5 THEN -- 5년이 지났다면
          SELECT CONCAT('데뷔한지 ', days, '일이나 지났습니다. 핑순이들 축하합니다!');
    ELSE
          SELECT '데뷔한지 ' + days + '일밖에 안되었네요. 핑순이들 화이팅~' ;
    END IF;
END $$
DELIMITER ;
CALL ifProc3();

SELECT CURRENT_DATE(), DATEDIFF('2021-12-31', '2000-1-1');

DROP PROCEDURE IF EXISTS caseProc; 
DELIMITER $$
CREATE PROCEDURE caseProc()
BEGIN
    DECLARE point INT ;
    DECLARE credit CHAR(1);
    SET point = 88 ;
    
    CASE 
        WHEN point >= 90 THEN
            SET credit = 'A';
        WHEN point >= 80 THEN
            SET credit = 'B';
        WHEN point >= 70 THEN
            SET credit = 'C';
        WHEN point >= 60 THEN
            SET credit = 'D';
        ELSE
            SET credit = 'F';
    END CASE;
    SELECT CONCAT('취득점수==>', point), CONCAT('학점==>', credit);
END $$
DELIMITER ;
CALL caseProc();


SELECT mem_id, SUM(price*amount) "총구매액"
   FROM buy
   GROUP BY mem_id;
   
SELECT mem_id, SUM(price*amount) "총구매액"
   FROM buy
   GROUP BY mem_id
   ORDER BY SUM(price*amount) DESC ;

SELECT B.mem_id, M.mem_name, SUM(price*amount) "총구매액"
   FROM buy B
         INNER JOIN member M
         ON B.mem_id = M.mem_id
   GROUP BY B.mem_id
   ORDER BY SUM(price*amount) DESC ;


SELECT M.mem_id, M.mem_name, SUM(price*amount) "총구매액"
   FROM buy B
         RIGHT OUTER JOIN member M
         ON B.mem_id = M.mem_id
   GROUP BY M.mem_id
   ORDER BY SUM(price*amount) DESC ;
   

SELECT M.mem_id, M.mem_name, SUM(price*amount) "총구매액",
        CASE  
           WHEN (SUM(price*amount)  >= 1500) THEN '최우수고객'
           WHEN (SUM(price*amount)  >= 1000) THEN '우수고객'
           WHEN (SUM(price*amount) >= 1 ) THEN '일반고객'
           ELSE '유령고객'
        END "회원등급"
   FROM buy B
         RIGHT OUTER JOIN member M
         ON B.mem_id = M.mem_id
   GROUP BY M.mem_id
   ORDER BY SUM(price*amount) DESC ;
   
DROP PROCEDURE IF EXISTS whileProc; 
DELIMITER $$
CREATE PROCEDURE whileProc()
BEGIN
    DECLARE i INT; -- 1에서 100까지 증가할 변수
    DECLARE hap INT; -- 더한 값을 누적할 변수
    SET i = 1;
    SET hap = 0;

    WHILE (i <= 100) DO
        SET hap = hap + i;  -- hap의 원래의 값에 i를 더해서 다시 hap에 넣으라는 의미
        SET i = i + 1;      -- i의 원래의 값에 1을 더해서 다시 i에 넣으라는 의미
    END WHILE;

    SELECT '1부터 100까지의 합 ==>', hap;   
END $$
DELIMITER ;
CALL whileProc();


DROP PROCEDURE IF EXISTS whileProc2; 
DELIMITER $$
CREATE PROCEDURE whileProc2()
BEGIN
    DECLARE i INT; -- 1에서 100까지 증가할 변수
    DECLARE hap INT; -- 더한 값을 누적할 변수
    SET i = 1;
    SET hap = 0;

    myWhile: 
    WHILE (i <= 100) DO  -- While문에 label을 지정
       IF (i%4 = 0) THEN
         SET i = i + 1;     
         ITERATE myWhile; -- 지정한 label문으로 가서 계속 진행
       END IF;
       SET hap = hap + i; 
       IF (hap > 1000) THEN 
         LEAVE myWhile; -- 지정한 label문을 떠남. 즉, While 종료.
       END IF;
       SET i = i + 1;
    END WHILE;

    SELECT '1부터 100까지의 합(4의 배수 제외), 1000 넘으면 종료 ==>', hap; 
END $$
DELIMITER ;
CALL whileProc2();

# 연습 코딩
drop procedure if exists whileProc2;
drop procedure if exists whileproc2;
delimiter $$
create procedure whileproc2()
begin 
declare i int;
declare cumm int;
set i = 1;
set cumm = 0;

mywhile:
while ( i <= 100) DO
if (i%4 = 0) THEN
set i = i+1;
iterate mywhile;
end if;
set cumm = cumm +i;
if (cumm >1000) THEN
leave mywhile;
end if;
set i = i+1;
end while;

select '1부터 100까지의 합(4의 배수 제외), 1000 넘으면 종료 ==>', cumm;
end $$
delimiter ;
call whileproc2

# 동적 코딩
# 매실행마다 값이 달라지는 것에 대응 가능함
use market_db;
PREPARE myQuery FROM 'SELECT * FROM member WHERE mem_id = "BLK"';
EXECUTE myQuery;
DEALLOCATE PREPARE myQuery;


DROP TABLE IF EXISTS gate_table;
CREATE TABLE gate_table (id INT AUTO_INCREMENT PRIMARY KEY, entry_time DATETIME);

SET @curDate = CURRENT_TIMESTAMP(); -- 현재 날짜와 시간

# values로 null과 ?과 준비돼있지만 execute로 실행하면, auto_increment랑 curdate덕분에 값이 생성됨됨
PREPARE myQuery FROM 'INSERT INTO gate_table VALUES(NULL, ?)';
EXECUTE myQuery USING @curDate;
DEALLOCATE PREPARE myQuery;

SELECT * FROM gate_table;

