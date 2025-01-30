-- ------------------------------
-- 1절
-- ------------------------------

CREATE DATABASE naver_db;

CREATE TABLE sample_table (num INT);

DROP DATABASE IF EXISTS naver_db;
CREATE DATABASE naver_db;

USE naver_db;
DROP TABLE IF EXISTS member;  -- 기존에 있으면 삭제
CREATE TABLE member -- 회원 테이블
( mem_id        CHAR(8), -- 회원 아이디(PK)
  mem_name      VARCHAR(10), -- 이름
  mem_number    TINYINT,  -- 인원수
  addr          CHAR(2), -- 주소(경기,서울,경남 식으로 2글자만입력)
  phone1        CHAR(3), -- 연락처의 국번(02, 031, 055 등)
  phone2        CHAR(8), -- 연락처의 나머지 전화번호(하이픈제외)
  height        TINYINT UNSIGNED,  -- 평균 키
  debut_date    DATE  -- 데뷔 일자
);

DROP TABLE IF EXISTS member;  -- 기존에 있으면 삭제
CREATE TABLE member -- 회원 테이블
( mem_id        CHAR(8) NOT NULL,
  mem_name      VARCHAR(10) NOT NULL, 
  mem_number    TINYINT NOT NULL, 
  addr          CHAR(2) NOT NULL,
  phone1        CHAR(3) NULL,
  phone2        CHAR(8) NULL,
  height        TINYINT UNSIGNED NULL, 
  debut_date    DATE NULL
);

DROP TABLE IF EXISTS member;  -- 기존에 있으면 삭제
CREATE TABLE member -- 회원 테이블
( mem_id        CHAR(8) NOT NULL PRIMARY KEY,
  mem_name      VARCHAR(10) NOT NULL, 
  mem_number    TINYINT NOT NULL, 
  addr          CHAR(2) NOT NULL,
  phone1        CHAR(3) NULL,
  phone2        CHAR(8) NULL,
  height        TINYINT UNSIGNED NULL, 
  debut_date    DATE NULL
);

DROP TABLE IF EXISTS buy;  -- 기존에 있으면 삭제
CREATE TABLE buy -- 구매 테이블
(  num         INT AUTO_INCREMENT NOT NULL PRIMARY KEY, -- 순번(PK)
   mem_id      CHAR(8) NOT NULL, -- 아이디(FK)
   prod_name     CHAR(6) NOT NULL, --  제품이름
   group_name     CHAR(4) NULL , -- 분류
   price         INT UNSIGNED NOT NULL, -- 가격
   amount        SMALLINT UNSIGNED  NOT NULL -- 수량
);

DROP TABLE IF EXISTS buy;  -- 기존에 있으면 삭제
CREATE TABLE buy 
(  num         INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
   mem_id      CHAR(8) NOT NULL, 
   prod_name     CHAR(6) NOT NULL, 
   group_name     CHAR(4) NULL ,
   price         INT UNSIGNED NOT NULL,
   amount        SMALLINT UNSIGNED  NOT NULL ,
   FOREIGN KEY(mem_id) REFERENCES member(mem_id) # 반드시 refer걸어주는 테이블의 변수가 pk여야 함. 아니면 에러 발생
);

INSERT INTO member VALUES('TWC', '트와이스', 9, '서울', '02', '11111111', 167, '2015-10-19');
INSERT INTO member VALUES('BLK', '블랙핑크', 4, '경남', '055', '22222222', 163, '2016-8-8');
INSERT INTO member VALUES('WMN', '여자친구', 6, '경기', '031', '33333333', 166, '2015-1-15');

INSERT INTO buy VALUES( NULL, 'BLK', '지갑', NULL, 30, 2);
INSERT INTO buy VALUES( NULL, 'BLK', '맥북프로', '디지털', 1000, 1);
INSERT INTO buy VALUES( NULL, 'APN', '아이폰', '디지털', 200, 1); # 앞서 만든 member 테이블 mem_id를 참조하고 있음. 근데 member의 mem_id에는 apn이 없음. 그래서 값 안들어감


-- ------------------------------
-- 2절
-- ------------------------------
# 제약조건 종류
USE naver_db;
DROP TABLE IF EXISTS buy, member;
CREATE TABLE member 
( mem_id  CHAR(8) NOT NULL PRIMARY KEY, 
  mem_name    VARCHAR(10) NOT NULL, 
  height      TINYINT UNSIGNED NULL
);

DESCRIBE member;
desc member;

DROP TABLE IF EXISTS member;
CREATE TABLE member 
( mem_id  CHAR(8) NOT NULL, 
  mem_name    VARCHAR(10) NOT NULL, 
  height      TINYINT UNSIGNED NULL,
  PRIMARY KEY (mem_id) # 꼭 변수명 옆에 안해도 됨. 깔끔하게 밑에 해도 됨
);

DROP TABLE IF EXISTS member;
CREATE TABLE member 
( mem_id  CHAR(8) NOT NULL, 
  mem_name    VARCHAR(10) NOT NULL, 
  height      TINYINT UNSIGNED NULL
);
ALTER TABLE member # 꼭 테이블 만들때 한번에 안하고, alter구문을 사용해서 추가해도 됨
     ADD CONSTRAINT 
     PRIMARY KEY (mem_id);

DROP TABLE IF EXISTS member;
CREATE TABLE member 
( mem_id  CHAR(8) NOT NULL, 
  mem_name    VARCHAR(10) NOT NULL, 
  height      TINYINT UNSIGNED NULL,
  CONSTRAINT PRIMARY KEY PK_member_mem_id (mem_id)
);

# 외래키 제약조건
DROP TABLE IF EXISTS buy, member;
CREATE TABLE member 
( mem_id  CHAR(8) NOT NULL PRIMARY KEY, 
  mem_name    VARCHAR(10) NOT NULL, 
  height      TINYINT UNSIGNED NULL
);
CREATE TABLE buy 
(  num         INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
   mem_id      CHAR(8) NOT NULL, 
   prod_name     CHAR(6) NOT NULL, 
   FOREIGN KEY(mem_id) REFERENCES member(mem_id)
);

# 변수명이 달라도 되긴 함
DROP TABLE IF EXISTS buy, member;
CREATE TABLE member 
( mem_id  CHAR(8) NOT NULL PRIMARY KEY, 
  mem_name    VARCHAR(10) NOT NULL, 
  height      TINYINT UNSIGNED NULL
);
CREATE TABLE buy 
(  num         INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
   user_id      CHAR(8) NOT NULL, 
   prod_name     CHAR(6) NOT NULL, 
   FOREIGN KEY(user_id) REFERENCES member(mem_id)
);

# alter로 제약주는 것도 가능

DROP TABLE IF EXISTS buy;
CREATE TABLE buy 
(  num         INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
   mem_id      CHAR(8) NOT NULL, 
   prod_name     CHAR(6) NOT NULL
);
ALTER TABLE buy
    ADD CONSTRAINT 
    FOREIGN KEY(mem_id) REFERENCES member(mem_id);

# 기존의 값 변경: update
INSERT INTO member VALUES('BLK', '블랙핑크', 163);
INSERT INTO buy VALUES(NULL, 'BLK', '지갑');
INSERT INTO buy VALUES(NULL, 'BLK', '맥북');

SELECT M.mem_id, M.mem_name, B.prod_name 
   FROM buy B
      INNER JOIN member M
      ON B.mem_id = M.mem_id;
      

# pk로 지정된 변수는 수정 또는 삭제 안됨. 물론 pk랑 관계 맺어진 변수도 제약
UPDATE member SET mem_id = 'PINK' WHERE mem_id='BLK';

DELETE FROM member WHERE  mem_id='BLK';

# pk변수를 꼭 수정하고 싶을 때,
# pk변수와 fk를 같이 수정되게 하면 가능
# 이 때 쓰는게 on update cascade, 또는 on delete cascade
DROP TABLE IF EXISTS buy;
CREATE TABLE buy 
(  num         INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
   mem_id      CHAR(8) NOT NULL, 
   prod_name     CHAR(6) NOT NULL
);
ALTER TABLE buy
    ADD CONSTRAINT 
    FOREIGN KEY(mem_id) REFERENCES member(mem_id)
    ON UPDATE CASCADE
    ON DELETE CASCADE;
    
INSERT INTO buy VALUES(NULL, 'BLK', '지갑');
INSERT INTO buy VALUES(NULL, 'BLK', '맥북');

SELECT * FROM member;

UPDATE member SET mem_id = 'PINK' WHERE mem_id='BLK';

SELECT M.mem_id, M.mem_name, B.prod_name 
   FROM buy B
      INNER JOIN member M
      ON B.mem_id = M.mem_id;

DELETE FROM member WHERE  mem_id='PINK';

SELECT * FROM buy;

# unique 제약조건
DROP TABLE IF EXISTS buy, member;
CREATE TABLE member 
( mem_id  CHAR(8) NOT NULL PRIMARY KEY, 
  mem_name    VARCHAR(10) NOT NULL, 
  height      TINYINT UNSIGNED NULL,
  email       CHAR(30)  NULL UNIQUE
);

INSERT INTO member VALUES('BLK', '블랙핑크', 163, 'pink@gmail.com');
INSERT INTO member VALUES('TWC', '트와이스', 167, NULL);
INSERT INTO member VALUES('APN', '에이핑크', 164, 'pink@gmail.com'); # 이메일이 겹쳐서 고유키 제약조건에 위배
SELECT * FROM member;

# 체크 제약조건, 조건에 만족하는 값만 입력되게 하는 제약
DROP TABLE IF EXISTS member;
CREATE TABLE member 
( mem_id  CHAR(8) NOT NULL PRIMARY KEY, 
  mem_name    VARCHAR(10) NOT NULL, 
  height      TINYINT UNSIGNED NULL CHECK (height >= 100),
  phone1      CHAR(3)  NULL
);

INSERT INTO member VALUES('BLK', '블랙핑크', 163, NULL);
INSERT INTO member VALUES('TWC', '트와이스', 99, NULL); # 조건 100이상을 충족하지 않아 입력x

ALTER TABLE member
    ADD CONSTRAINT
    CHECK  (phone1 IN ('02', '031', '032', '054', '055', '061' )) ;

INSERT INTO member VALUES('TWC', '트와이스', 167, '02');
INSERT INTO member VALUES('OMY', '오마이걸', 167, '010'); # 값 해당 범위에 없음, 조건 위배

# 기본값 정의
DROP TABLE IF EXISTS member;
CREATE TABLE member 
( mem_id  CHAR(8) NOT NULL PRIMARY KEY, 
  mem_name    VARCHAR(10) NOT NULL, 
  height      TINYINT UNSIGNED NULL DEFAULT 160,
  phone1      CHAR(3)  NULL
);

ALTER TABLE member
    ALTER COLUMN phone1 SET DEFAULT '02';

INSERT INTO member VALUES('RED', '레드벨벳', 161, '054');
INSERT INTO member VALUES('SPC', '우주소녀', default, default);
INSERT INTO member VALUES('aaa', '건전지', null, null);
SELECT * FROM member;


-- ------------------------------
-- 3절
-- ------------------------------
# 뷰 생성
USE market_db;
SELECT mem_id, mem_name, addr FROM member;

USE market_db;
CREATE VIEW v_member # 뷰 먼저 만들고 테이블도 형성. 뷰　테이블이라고　ｖ＿사용
AS
    SELECT mem_id, mem_name, addr FROM member;
# 이건 뷰에 있는 가상의 테이블
SELECT * FROM v_member;

drop view if exists v_member;

create view v_member
as select mem_id, mem_name, addr from member;

select * from v_member;

SELECT mem_name, addr FROM v_member
   WHERE addr IN ('서울', '경기');

SELECT B.mem_id, M.mem_name, B.prod_name, M.addr, 
        CONCAT(M.phone1, M.phone2) '연락처' 
   FROM buy B
     INNER JOIN member M
     ON B.mem_id = M.mem_id;

CREATE VIEW v_memberbuy
AS
    SELECT B.mem_id, M.mem_name, B.prod_name, M.addr, 
            CONCAT(M.phone1, M.phone2) '연락처' 
       FROM buy B
         INNER JOIN member M
         ON B.mem_id = M.mem_id;

SELECT * FROM v_memberbuy WHERE mem_name = '블랙핑크';

# 별칭 사용
USE market_db;
CREATE VIEW v_viewtest1
AS
    SELECT B.mem_id 'Member ID', M.mem_name AS 'Member Name', 
            B.prod_name "Product Name",  
            CONCAT(M.phone1, M.phone2) AS "Office Phone" 
       FROM buy B
         INNER JOIN member M
         ON B.mem_id = M.mem_id;
         
SELECT  DISTINCT `Member ID`, `Member Name` FROM v_viewtest1; -- 백틱을 사용

ALTER VIEW v_viewtest1
AS
    SELECT B.mem_id '회원 아이디', M.mem_name AS '회원 이름', 
            B.prod_name "제품 이름", 
            CONCAT(M.phone1, M.phone2) AS "연락처" 
       FROM buy B
         INNER JOIN member M
         ON B.mem_id = M.mem_id;
         
SELECT  DISTINCT `회원 아이디`, `회원 이름` FROM v_viewtest1;  -- 별칭으로 만든 이름 띄워쓰기가 포함된 경우 키보드 1옆에 있는 ` 백틱을 사용

DROP VIEW v_viewtest1;

USE market_db;
CREATE OR REPLACE VIEW v_viewtest2
AS
    SELECT mem_id, mem_name, addr FROM member;

DESCRIBE v_viewtest2;

DESCRIBE member;

SHOW CREATE VIEW v_viewtest2;

UPDATE v_member SET addr = '부산' WHERE mem_id='BLK' ;
select * from v_member;

# 안됨, 보이는거는 3개지만, 실제 진행 과정은 입력된 쿼리를 실제 테이블에 적용한 걸 뷰 테이블에 보여주는 것이기 때문에
# 수정하려면 원래 테이블(member)에 적용될 수 있게 코드를 작성해야함
# 원래 테이블에 인원같은건 not null이라서 그 조건에 맞춰서 해야 함
INSERT INTO v_member(mem_id, mem_name, addr) VALUES('BTS','방탄소년단','경기') ;

CREATE VIEW v_height167
AS
    SELECT * FROM member WHERE height >= 167 ;
    
SELECT * FROM v_height167 ;

# 167이상만 모여있는 뷰 테이블이라 삭제되는 행 없음
DELETE FROM v_height167 WHERE height < 167;

# 보이지 않는 변수의 데이터 제약 조건도 모두 만족하게 값은 넣었지만, 뷰 테이블의 조건 167 이상의 조건은 만족하지 않음
INSERT INTO v_height167 VALUES('TRA','티아라', 6, '서울', NULL, NULL, 159, '2005-01-01') ;

SELECT * FROM v_height167;

ALTER VIEW v_height167
AS
    SELECT * FROM member WHERE height >= 167
        WITH CHECK OPTION ; # 이 조건에 맞는 값만 입력되게 확실하게 조건을 넣은거
        
# 애초에 with check option으로 제약주니까, 데이터 입력 최소 조건은 만족했지만 167조건은 만족하지 않아 에러 발생
INSERT INTO v_height167 VALUES('TOB','텔레토비', 4, '영국', NULL, NULL, 140, '1995-01-01') ;

CREATE VIEW v_complex
AS
    SELECT B.mem_id, M.mem_name, B.prod_name, M.addr
        FROM buy B
            INNER JOIN member M
            ON B.mem_id = M.mem_id;

DROP TABLE IF EXISTS buy, member;

# 뷰가 참조하는 테이블이 없어져서, 뷰도 실행안됨, 뷰가 참조하고 있다고 해서 테이블이 없어지지는 않음
SELECT * FROM v_height167;

# 뷰가 왜 안되는지 확인하려면
CHECK TABLE v_height167;

