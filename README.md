# Vision Inspection

디스플레이 불량검사 업무를 하면서 경험했던 영상처리와
최근 공부하고 있는 딥러닝을 연결해보려고 만든 개인 프로젝트입니다.

기존에는 C++과 MFC를 중심으로 검사 프로그램을 개발했지만,
이번 프로젝트에서는 OpenCV로 기본 검사를 만들고
Python과 PyTorch로 딥러닝 검사도 추가해보려고 합니다.

## 만들고 싶은 것

이미지를 입력하면 다음 순서로 검사하는 프로그램을 만들 예정입니다.

1. C++과 OpenCV로 기본 영상처리 검사
2. Python과 PyTorch로 딥러닝 불량 검사
3. 두 검사 결과 비교
4. 불량 위치와 confidence 저장
5. 이미지별 처리 시간 기록
6. 필요하면 검사 결과를 다시 확인하거나 재검사

## 현재 진행 상황

- [x] Git 저장소 생성
- [x] GitHub 연결
- [x] README 작성
- [ ] 검사할 불량 유형 결정
- [ ] 테스트 이미지 준비
- [ ] C++ OpenCV 기본 검사 구현
- [ ] Python 딥러닝 모델 구현
- [ ] 두 검사 결과 통합
- [ ] 처리 시간과 검사 결과 비교
- [ ] AI Agent 연결

## 사용할 기술

- C++
- OpenCV
- Python
- PyTorch
- Git


# Dataset

Carinthia SEM Defect Dataset을 사용합니다.

Source:
https://zenodo.org/records/10696644

원본 데이터와 압축 파일은 용량과 라이선스 문제로 Git 저장소에 포함하지 않습니다.
필요한 경우 위 출처에서 직접 다운로드합니다.