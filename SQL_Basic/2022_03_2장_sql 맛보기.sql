select * from member;
select * from member where member_name = '아이유';

# 인덱스로 찾기
create index idx_member_name on member(member_name);
select * from member;

# 뷰: 바로가기 아이콘, 가상화면 눌러서 바로 보이게
create view member_view 
as
select * from member; # 뷰 만들기 세미콜론으로 끝에 닫아서 한 문장으로 만듦

select * from member_view;

# 스토어드: 일반 프로그래밍 언어와 비슷하게 조건문 등 사용할 수 있는 코드
# 원하는 조건문을 저장해서 한번에 사용
delimiter //
create procedure myProc()
begin
select * from member where member_name = '나훈아';
select * from product where product_name = '삼각김밥';
end//
delimiter  ; 

call myProc();
