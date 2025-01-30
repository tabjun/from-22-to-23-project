-- ------------------------------
-- 1절
-- ------------------------------

USE market_db;
CREATE TABLE table1  (
    col1  INT  PRIMARY KEY,
    col2  INT,
    col3  INT
);
SHOW INDEX FROM table1;

CREATE TABLE table2  (
    col1  INT  PRIMARY KEY,
    col2  INT  UNIQUE,
    col3  INT  UNIQUE
);
SHOW INDEX FROM table2;

USE market_db;
DROP TABLE IF EXISTS buy, member;
CREATE TABLE member 
( mem_id      CHAR(8) , 
  mem_name    VARCHAR(10),
  mem_number  INT ,  
  addr        CHAR(2)  
 );

INSERT INTO member VALUES('TWC', '트와이스', 9, '서울');
INSERT INTO member VALUES('BLK', '블랙핑크', 4, '경남');
INSERT INTO member VALUES('WMN', '여자친구', 6, '경기');
INSERT INTO member VALUES('OMY', '오마이걸', 7, '서울');
SELECT * FROM member;

alter table member
add constraint
primary key (mem_id);
# 클러스터형 인덱스는 자동 생성 및 정렬
# mem_id를 기준으로 오름차순 됨
SELECT * FROM member;

ALTER TABLE member DROP PRIMARY KEY ; -- 기본 키 제거
ALTER TABLE member 
    ADD CONSTRAINT 
    PRIMARY KEY(mem_name);
SELECT * FROM member;

# mem_name을 기준으로 오름차순 정렬됨
# 소녀시대가 입력된 게 마지막에 입력되지 않고 오름차순으로 정렬돼서 결과 출력력
INSERT INTO member VALUES('GRL', '소녀시대', 8, '서울');
SELECT * FROM member;


USE market_db;
DROP TABLE IF EXISTS member;
CREATE TABLE member 
( mem_id      CHAR(8) , 
  mem_name    VARCHAR(10),
  mem_number  INT ,  
  addr        CHAR(2)  
 );

INSERT INTO member VALUES('TWC', '트와이스', 9, '서울');
INSERT INTO member VALUES('BLK', '블랙핑크', 4, '경남');
INSERT INTO member VALUES('WMN', '여자친구', 6, '경기');
INSERT INTO member VALUES('OMY', '오마이걸', 7, '서울');
SELECT * FROM member;

ALTER TABLE member
     ADD CONSTRAINT 
     UNIQUE (mem_id);
SELECT * FROM member;

ALTER TABLE member
     ADD CONSTRAINT 
     UNIQUE (mem_name);
SELECT * FROM member;

INSERT INTO member VALUES('GRL', '소녀시대', 8, '서울');
SELECT * FROM member;

-- ------------------------------
-- 2절
-- ------------------------------

# 노드, 페이지라고도 부름
# 의사 결정 나무 구조랑 같긴 함

USE market_db;
CREATE TABLE cluster  -- 클러스터형 테이블 
( mem_id      CHAR(8) , 
  mem_name    VARCHAR(10)
 );
INSERT INTO cluster VALUES('TWC', '트와이스');
INSERT INTO cluster VALUES('BLK', '블랙핑크');
INSERT INTO cluster VALUES('WMN', '여자친구');
INSERT INTO cluster VALUES('OMY', '오마이걸');
INSERT INTO cluster VALUES('GRL', '소녀시대');
INSERT INTO cluster VALUES('ITZ', '잇지');
INSERT INTO cluster VALUES('RED', '레드벨벳');
INSERT INTO cluster VALUES('APN', '에이핑크');
INSERT INTO cluster VALUES('SPC', '우주소녀');
INSERT INTO cluster VALUES('MMU', '마마무');

SELECT * FROM cluster;

ALTER TABLE cluster
    ADD CONSTRAINT 
    PRIMARY KEY (mem_id);

SELECT * FROM cluster;


USE market_db;
CREATE TABLE second  -- 보조 인덱스 테이블 
( mem_id      CHAR(8) , 
  mem_name    VARCHAR(10)
 );
INSERT INTO second VALUES('TWC', '트와이스');
INSERT INTO second VALUES('BLK', '블랙핑크');
INSERT INTO second VALUES('WMN', '여자친구');
INSERT INTO second VALUES('OMY', '오마이걸');
INSERT INTO second VALUES('GRL', '소녀시대');
INSERT INTO second VALUES('ITZ', '잇지');
INSERT INTO second VALUES('RED', '레드벨벳');
INSERT INTO second VALUES('APN', '에이핑크');
INSERT INTO second VALUES('SPC', '우주소녀');
INSERT INTO second VALUES('MMU', '마마무');

ALTER TABLE second
    ADD CONSTRAINT 
    UNIQUE (mem_id);

SELECT * FROM second;


-- ------------------------------
-- 3절
-- ------------------------------

# 인덱스를 제거할 때 drop으로 제거하면 됨
# pk나 unique 같은 기본키는 설정하면 인덱스가 자동으로 생성됨. 이 경우에는 drop index로 제거 불가. 키 제거하면 인덱스도 같이 제거됨 
USE market_db;
SELECT * FROM member;

# 인덱스 확인
SHOW INDEX FROM member;

SHOW TABLE STATUS LIKE 'member';

CREATE INDEX idx_member_addr 
   ON member (addr);

# non unique=1 이면 중복값이 있음 1=yes
SHOW INDEX FROM member;

SHOW TABLE STATUS LIKE 'member';

# 인덱스를 만들어주고 alalyze로 적용해줘야 함
ANALYZE TABLE member;
SHOW TABLE STATUS LIKE 'member';

CREATE UNIQUE INDEX idx_member_mem_number
    ON member (mem_number); -- 오류 발생

select * from member;
CREATE INDEX idx_member_mem_name
    ON member (mem_name);

SHOW INDEX FROM member;

# unique 인덱스로 만들어서 마마무가 중복되기 때문에 에러 발생
INSERT INTO member VALUES('MOO', '마마무', 2, '태국', '001', '12341234', 155, '2020.10.10');

ANALYZE TABLE member;  -- 지금까지 만든 인덱스를 모두 적용
SHOW INDEX FROM member;


SELECT * FROM member;

SELECT mem_id, mem_name, addr FROM member;

SELECT mem_id, mem_name, addr 
    FROM member 
    WHERE mem_name = '에이핑크';
    
    
CREATE INDEX idx_member_mem_number
    ON member (mem_number);
ANALYZE TABLE member; -- 인덱스 적용

# 이건 인덱스 방식으로 찾음
SELECT mem_name, mem_number 
    FROM member 
    WHERE mem_number >= 7; 
    
# 이건 인덱스 방식으로 찾지 않음, 왜냐하면 mysql에서 자체적으로 1보다 큰거 찾는거는 그냥 다 찾는게 좋다고 판단
SELECT mem_name, mem_number 
    FROM member 
    WHERE mem_number >= 1; 
    
# 이건 7이상인거랑 똑같은 방식이라 인덱스랑 같을 것 같지만, 전체 찾기로 찾음. 인덱스 찾기 방식으로 찾으려면 where절에서 연산자를 사용하지 말아야 함
# 물론 결과는 같게 나오지만 데이터의 양이 많을 경우 인덱스 찾기 방식이 더 빠름
SELECT mem_name, mem_number 
    FROM member 
    WHERE mem_number*2 >= 14;     
    
SELECT mem_name, mem_number 
    FROM member 
    WHERE mem_number >= 14/2;   
    
SHOW INDEX FROM member;

DROP INDEX idx_member_mem_name ON member;
DROP INDEX idx_member_addr ON member;
DROP INDEX idx_member_mem_number ON member;

# 만약 pk를 참조하는 fk가 있는 경우, fk를 먼저 제거해야 pk제거 가능
alter table member
drop primary key;

# 해당 db의 외래키 확인하는 코드
SELECT table_name, constraint_name
    FROM information_schema.referential_constraints
    WHERE constraint_schema = 'market_db';

ALTER TABLE buy 
    DROP FOREIGN KEY buy_ibfk_1;
ALTER TABLE member 
    DROP PRIMARY KEY;

    
SELECT mem_id, mem_name, mem_number, addr 
    FROM member 
    WHERE mem_name = '에이핑크';