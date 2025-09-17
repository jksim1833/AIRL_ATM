#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TETRIS 자동화 테스트 시스템 v3.0
=================================
최적화된 이미지 기반 시나리오 테스트 프레임워크

Author: AI Research Lab
Version: 3.0.0
License: MIT
"""

import sys
import json
import base64
import logging
import asyncio
from pathlib import Path
from time import perf_counter
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from contextlib import contextmanager


# ==================== Configuration ====================
@dataclass
class TestConfig:
    """테스트 설정 관리"""
    people_count: int = 2
    trials: int = 1
    timeout: float = 120.0
    parallel_workers: int = 1
    save_individual_results: bool = True
    save_summary: bool = True
    verbose: bool = False

@dataclass
class TestResult:
    """개별 테스트 결과"""
    image_name: str
    trial: int
    success: bool
    execution_time: float
    error: Optional[str] = None
    chain1_out: Optional[str] = None
    chain2_out: Optional[str] = None
    chain3_out: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


# ==================== Core Test Engine ====================
class TetrisTestEngine:
    """고성능 테트리스 테스트 엔진"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.paths = self._initialize_paths()
        self.main_chain = self._load_main_chain()
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 시스템 초기화"""
        logger = logging.getLogger('TetrisTest')
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_paths(self) -> Dict[str, Path]:
        """경로 초기화 및 검증"""
        base_path = Path(__file__).resolve().parent
        
        paths = {
            'base': base_path,
            'tetris': base_path / 'tetris',
            'main_chain': base_path / 'tetris' / 'main_chain',
            'user_input': base_path / 'tetris' / 'user_input',
            'images': base_path / 'chain1_image',
            'results': base_path / 'result'
        }
        
        # 필수 경로 검증
        required_paths = ['tetris', 'main_chain', 'user_input', 'images']
        missing_paths = []
        
        for key in required_paths:
            path = paths[key]
            if not path.exists():
                missing_paths.append(f"{key}: {path}")
                
        if missing_paths:
            raise FileNotFoundError(
                f"필수 디렉토리가 없습니다:\n" + 
                "\n".join(f"  - {p}" for p in missing_paths)
            )
        
        # 결과 디렉토리 생성
        paths['results'].mkdir(exist_ok=True)
        
        # Python 경로에 추가
        for path_key in ['main_chain', 'user_input', 'tetris']:
            path_str = str(paths[path_key])
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        
        self.logger.info(f"경로 초기화 완료: {len(paths)}개 경로 설정")
        return paths
    
    def _load_main_chain(self):
        """main_chain 모듈 로드"""
        try:
            import main_chain as MC
            self.logger.info("✓ main_chain 모듈 로드 성공")
            return MC
        except ImportError as e:
            self.logger.error(f"main_chain 모듈 로드 실패: {e}")
            raise
        except Exception as e:
            self.logger.error(f"예상치 못한 오류: {e}")
            raise
    
    def get_image_files(self) -> List[Path]:
        """이미지 파일 목록 획득"""
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for pattern in patterns:
            image_files.extend(self.paths['images'].glob(pattern))
        
        # 자연 정렬 (1.jpg, 2.jpg, ..., 10.jpg)
        image_files.sort(key=lambda x: (len(x.stem), x.stem.lower()))
        
        self.logger.info(f"이미지 파일 {len(image_files)}개 발견")
        return image_files
    
    @staticmethod
    def image_to_data_url(image_path: Path) -> str:
        """이미지를 data URL로 변환 (최적화)"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # MIME 타입 자동 감지
        mime_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.bmp': 'image/bmp',
            '.tiff': 'image/tiff', '.tif': 'image/tiff'
        }
        
        ext = image_path.suffix.lower()
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        encoded = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{encoded}"
    
    def execute_single_test(self, image_path: Path, trial: int, save_immediately: bool = True) -> TestResult:
        """단일 테스트 실행 (최적화) - 즉시 파일 저장 옵션 포함"""
        start_time = perf_counter()
        image_name = image_path.stem
        
        try:
            # 이미지 변환
            image_data_url = self.image_to_data_url(image_path)
            
            # main_chain 입력 생성
            user_msgs = self.main_chain.make_chain1_user_input(
                people_count=self.config.people_count,
                image_data_url=image_data_url
            )
            
            # 체인 실행
            chain_start = perf_counter()
            result = self.main_chain.seq_chain.invoke({
                "user_input": user_msgs,
                "people_count": self.config.people_count
            })
            execution_time = perf_counter() - chain_start
            
            test_result = TestResult(
                image_name=image_name,
                trial=trial,
                success=True,
                execution_time=execution_time,
                chain1_out=result.get("chain1_out", ""),
                chain2_out=result.get("chain2_out", ""),
                chain3_out=result.get("chain3_out", "")
            )
            
            # 즉시 개별 파일 저장
            if save_immediately and self.config.save_individual_results:
                self._save_single_result_immediately(test_result)
            
            return test_result
            
        except Exception as e:
            execution_time = perf_counter() - start_time
            self.logger.error(f"테스트 실패 [{image_name}-{trial}]: {str(e)}")
            
            test_result = TestResult(
                image_name=image_name,
                trial=trial,
                success=False,
                execution_time=execution_time,
                error=str(e)
            )
            
            # 실패한 경우에도 즉시 저장
            if save_immediately and self.config.save_individual_results:
                self._save_single_result_immediately(test_result)
            
            return test_result
    
    def _save_single_result_immediately(self, result: TestResult) -> None:
        """단일 결과를 즉시 파일로 저장"""
        filename = f"{result.image_name}_trial_{result.trial:02d}.txt"
        filepath = self.paths['results'] / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                self._write_individual_result_content(f, result)
            
            self.logger.debug(f"결과 파일 저장 완료: {filename}")
            
        except Exception as e:
            self.logger.error(f"결과 파일 저장 실패 [{filename}]: {e}")
    
    def _write_individual_result_content(self, file, result: TestResult) -> None:
        """개별 결과 파일 내용 작성 (공통 함수)"""
        file.write("=" * 60 + "\n")
        file.write(f"테스트 결과: {result.image_name} - Trial {result.trial}\n")
        file.write("=" * 60 + "\n")
        file.write(f"실행 시간: {result.timestamp}\n")
        file.write(f"이미지 파일: {result.image_name}\n")
        file.write(f"인원 수: {self.config.people_count}\n")
        file.write(f"실행 성공: {result.success}\n")
        file.write(f"실행 시간: {result.execution_time:.3f}초\n\n")
        
        if result.success:
            sections = [
                ("chain1_out", result.chain1_out),
                ("chain2_out", result.chain2_out), 
                ("chain3_out", result.chain3_out)
            ]
            
            for section_name, content in sections:
                file.write("=" * 20 + f"[ {section_name} ]" + "=" * 20 + "\n")
                file.write((content or "") + "\n\n")
        else:
            file.write("=" * 20 + "[ ERROR ]" + "=" * 20 + "\n")
            file.write(f"오류 내용: {result.error}\n")
    
    def run_parallel_tests(self, image_files: List[Path]) -> List[TestResult]:
        """병렬 테스트 실행 - 각 결과 즉시 저장 및 이미지별 중간 보고서 생성"""
        tasks = []
        image_results_tracker = {img.stem: [] for img in image_files}
        completed_images = set()
        
        # 테스트 태스크 생성
        for image_file in image_files:
            for trial in range(1, self.config.trials + 1):
                tasks.append((image_file, trial))
        
        results = []
        completed_count = 0
        total_tasks = len(tasks)
        
        self.logger.info(f"병렬 테스트 시작: {total_tasks}개 태스크, {self.config.parallel_workers}개 워커")
        self.logger.info(f"각 테스트 완료 시 개별 파일로 즉시 저장되며, 이미지별 중간 보고서도 생성됩니다.")
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # 태스크 제출 (즉시 저장 활성화)
            future_to_task = {
                executor.submit(self.execute_single_test, img, trial, True): (img.stem, trial, img)
                for img, trial in tasks
            }
            
            # 결과 수집
            for future in as_completed(future_to_task):
                img_name, trial, img_path = future_to_task[future]
                
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                    image_results_tracker[img_name].append(result)
                    completed_count += 1
                    
                    status = "✓" if result.success else "✗"
                    filename = f"{img_name}_trial_{trial:02d}.txt"
                    self.logger.info(
                        f"[{completed_count:3d}/{total_tasks}] {status} {img_name}-{trial:02d} "
                        f"({result.execution_time:.2f}s) → 저장: {filename}"
                    )
                    
                    # 현재 이미지의 모든 시행이 완료되었는지 확인
                    if (len(image_results_tracker[img_name]) == self.config.trials and 
                        img_name not in completed_images):
                        completed_images.add(img_name)
                        
                        # 중간 보고서 생성
                        if self.config.save_summary:
                            self._generate_intermediate_summary(img_name, image_results_tracker[img_name])
                    
                except Exception as e:
                    self.logger.error(f"태스크 실행 오류 [{img_name}-{trial}]: {e}")
                    error_result = TestResult(
                        image_name=img_name,
                        trial=trial,
                        success=False,
                        execution_time=0,
                        error=f"Execution timeout or error: {str(e)}"
                    )
                    
                    # 오류 결과도 즉시 저장
                    if self.config.save_individual_results:
                        self._save_single_result_immediately(error_result)
                    
                    results.append(error_result)
                    image_results_tracker[img_name].append(error_result)
                    completed_count += 1
                    
                    # 오류가 있어도 해당 이미지의 모든 시행이 완료되면 중간 보고서 생성
                    if (len(image_results_tracker[img_name]) == self.config.trials and 
                        img_name not in completed_images):
                        completed_images.add(img_name)
                        
                        if self.config.save_summary:
                            self._generate_intermediate_summary(img_name, image_results_tracker[img_name])
        
        self.logger.info(f"병렬 테스트 완료: 모든 결과가 개별 파일로 저장되고 이미지별 중간 보고서도 생성되었습니다.")
        return results
    
    def run_sequential_tests(self, image_files: List[Path]) -> List[TestResult]:
        """순차 테스트 실행 - 각 결과 즉시 저장 및 이미지별 중간 보고서 생성"""
        results = []
        total_tests = len(image_files) * self.config.trials
        current_test = 0
        
        self.logger.info(f"순차 테스트 시작: {total_tests}개 테스트")
        self.logger.info(f"각 테스트 완료 시 개별 파일로 즉시 저장됩니다.")
        
        for i, image_file in enumerate(image_files, 1):
            image_name = image_file.stem
            self.logger.info(f"[{i}/{len(image_files)}] 이미지: {image_name}")
            
            # 현재 이미지의 결과를 저장할 리스트
            current_image_results = []
            
            for trial in range(1, self.config.trials + 1):
                current_test += 1
                
                with self._progress_context(f"Trial {trial}/{self.config.trials}"):
                    # 즉시 저장 활성화
                    result = self.execute_single_test(image_file, trial, save_immediately=True)
                    results.append(result)
                    current_image_results.append(result)
                    
                    status = "✓" if result.success else "✗"
                    filename = f"{image_name}_trial_{trial:02d}.txt"
                    self.logger.info(
                        f"  {status} Trial {trial:02d} ({result.execution_time:.2f}s) → 저장: {filename}"
                    )
            
            # 현재 이미지의 모든 시행이 완료되면 중간 보고서 생성
            if self.config.save_summary:
                self._generate_intermediate_summary(image_name, current_image_results)
        
        self.logger.info(f"순차 테스트 완료: 모든 결과가 개별 파일로 저장되었습니다.")
        return results
    
    def _generate_intermediate_summary(self, image_name: str, image_results: List[TestResult]) -> None:
        """특정 이미지의 중간 보고서 생성"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_filename = f"summary_{image_name}_{timestamp}.txt"
            summary_filepath = self.paths['results'] / summary_filename
            
            # 통계 계산
            total_trials = len(image_results)
            successful_trials = [r for r in image_results if r.success]
            success_count = len(successful_trials)
            success_rate = (success_count / total_trials) * 100 if total_trials > 0 else 0
            
            execution_times = [r.execution_time for r in successful_trials]
            avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
            min_time = min(execution_times) if execution_times else 0
            max_time = max(execution_times) if execution_times else 0
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"이미지별 중간 보고서: {image_name}\n")
                f.write("=" * 70 + "\n")
                f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"이미지 파일: {image_name}\n")
                f.write(f"인원 수: {self.config.people_count}명\n\n")
                
                # 기본 통계
                f.write("=" * 30 + "[ 기본 통계 ]" + "=" * 30 + "\n")
                f.write(f"총 시행 횟수: {total_trials}회\n")
                f.write(f"성공한 시행: {success_count}회\n")
                f.write(f"실패한 시행: {total_trials - success_count}회\n")
                f.write(f"성공률: {success_rate:.1f}%\n\n")
                
                # 성능 통계
                if execution_times:
                    f.write("=" * 30 + "[ 성능 통계 ]" + "=" * 30 + "\n")
                    f.write(f"평균 실행 시간: {avg_time:.2f}초\n")
                    f.write(f"최단 실행 시간: {min_time:.2f}초\n")
                    f.write(f"최장 실행 시간: {max_time:.2f}초\n")
                    f.write(f"총 처리 시간: {sum(execution_times):.2f}초\n\n")
                
                # 시행별 상세 결과
                f.write("=" * 30 + "[ 시행별 결과 ]" + "=" * 30 + "\n")
                for result in image_results:
                    status = "성공" if result.success else "실패"
                    f.write(f"Trial {result.trial:02d}: {status} ({result.execution_time:.2f}초)")
                    if not result.success and result.error:
                        f.write(f" - {result.error}")
                    f.write("\n")
                
                # 실패 분석
                failed_results = [r for r in image_results if not r.success]
                if failed_results:
                    f.write(f"\n" + "=" * 30 + "[ 실패 분석 ]" + "=" * 30 + "\n")
                    error_counts = {}
                    for result in failed_results:
                        error_type = result.error if result.error else "Unknown"
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    
                    for error, count in error_counts.items():
                        f.write(f"- {error}: {count}회\n")
                
                # 성공한 경우의 결과 요약 (첫 번째 성공 결과만)
                if successful_trials:
                    first_success = successful_trials[0]
                    f.write(f"\n" + "=" * 25 + "[ 첫 번째 성공 결과 샘플 ]" + "=" * 25 + "\n")
                    f.write(f"Trial {first_success.trial}의 결과:\n\n")
                    
                    if first_success.chain1_out:
                        f.write("Chain1 결과 (요약):\n")
                        # 첫 200자만 표시
                        chain1_summary = first_success.chain1_out[:200] + "..." if len(first_success.chain1_out) > 200 else first_success.chain1_out
                        f.write(f"{chain1_summary}\n\n")
                    
                    if first_success.chain2_out:
                        f.write("Chain2 결과 (요약):\n")
                        chain2_summary = first_success.chain2_out[:200] + "..." if len(first_success.chain2_out) > 200 else first_success.chain2_out
                        f.write(f"{chain2_summary}\n\n")
                    
                    if first_success.chain3_out:
                        f.write("Chain3 결과 (요약):\n")
                        chain3_summary = first_success.chain3_out[:200] + "..." if len(first_success.chain3_out) > 200 else first_success.chain3_out
                        f.write(f"{chain3_summary}\n\n")
            
            self.logger.info(f"  중간 보고서 생성: {summary_filename}")
            
        except Exception as e:
            self.logger.error(f"중간 보고서 생성 실패 [{image_name}]: {e}")
    
    def _generate_missing_intermediate_summaries(self, all_results: List[TestResult]) -> None:
        """누락된 이미지별 중간 보고서 생성 (병렬 모드 보완용)"""
        # 이미지별로 결과 그룹화
        results_by_image = {}
        for result in all_results:
            if result.image_name not in results_by_image:
                results_by_image[result.image_name] = []
            results_by_image[result.image_name].append(result)
        
        # 각 이미지별로 중간 보고서가 있는지 확인하고 없으면 생성
        for image_name, image_results in results_by_image.items():
            if len(image_results) == self.config.trials:  # 해당 이미지의 모든 시행이 완료된 경우
                # 기존 중간 보고서 파일 확인
                pattern = f"summary_{image_name}_*.txt"
                existing_summaries = list(self.paths['results'].glob(pattern))
                
                if not existing_summaries:  # 중간 보고서가 없는 경우만 생성
                    self.logger.info(f"누락된 중간 보고서 생성: {image_name}")
                    self._generate_intermediate_summary(image_name, image_results)
    
    @contextmanager
    def _progress_context(self, description: str):
        """진행률 표시를 위한 컨텍스트 매니저"""
        if self.config.verbose:
            print(f"  {description}... ", end="", flush=True)
        yield
        if self.config.verbose:
            print("완료")


# ==================== Results Management ====================
class ResultsManager:
    """결과 관리 시스템"""
    
    def __init__(self, results_path: Path, verbose: bool = False):
        self.results_path = results_path
        self.verbose = verbose
        self.logger = logging.getLogger('TetrisTest.Results')
    
    def save_individual_results(self, results: List[TestResult]) -> None:
        """개별 결과 파일 저장 (이미 저장된 경우 스킵)"""
        if not self.config.save_individual_results:
            return
            
        self.logger.info("개별 결과 파일 저장 상태 확인 중...")
        
        saved_count = 0
        skipped_count = 0
        
        for result in results:
            filename = f"{result.image_name}_trial_{result.trial:02d}.txt"
            filepath = self.results_path / filename
            
            if filepath.exists():
                skipped_count += 1
                self.logger.debug(f"이미 저장됨: {filename}")
            else:
                # 파일이 없는 경우에만 저장 (누락된 결과 복구)
                with open(filepath, 'w', encoding='utf-8') as f:
                    self._write_individual_result(f, result)
                saved_count += 1
                self.logger.debug(f"누락된 결과 저장: {filename}")
        
        if saved_count > 0:
            self.logger.info(f"누락된 개별 결과 파일 {saved_count}개 저장 완료")
        if skipped_count > 0:
            self.logger.info(f"이미 저장된 파일 {skipped_count}개 확인")
    
    def _write_individual_result(self, file, result: TestResult) -> None:
        """개별 결과 파일 내용 작성 (ResultsManager용 - TetrisTestEngine과 동일한 형식)"""
        file.write("=" * 60 + "\n")
        file.write(f"테스트 결과: {result.image_name} - Trial {result.trial}\n")
        file.write("=" * 60 + "\n")
        file.write(f"실행 시간: {result.timestamp}\n")
        file.write(f"이미지 파일: {result.image_name}\n")
        file.write(f"실행 성공: {result.success}\n")
        file.write(f"실행 시간: {result.execution_time:.3f}초\n\n")
        
        if result.success:
            sections = [
                ("chain1_out", result.chain1_out),
                ("chain2_out", result.chain2_out),
                ("chain3_out", result.chain3_out)
            ]
            
            for section_name, content in sections:
                file.write("=" * 20 + f"[ {section_name} ]" + "=" * 20 + "\n")
                file.write((content or "") + "\n\n")
        else:
            file.write("=" * 20 + "[ ERROR ]" + "=" * 20 + "\n")
            file.write(f"오류 내용: {result.error}\n")
    
    def save_summary_report(self, results: List[TestResult], total_time: float) -> None:
        """요약 보고서 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 요약
        json_file = self.results_path / f"summary_{timestamp}.json"
        self._save_json_summary(json_file, results, total_time)
        
        # 텍스트 요약
        txt_file = self.results_path / f"summary_{timestamp}.txt"
        self._save_text_summary(txt_file, results, total_time)
        
        self.logger.info(f"요약 보고서 저장: {json_file.name}, {txt_file.name}")
    
    def _save_json_summary(self, filepath: Path, results: List[TestResult], total_time: float) -> None:
        """JSON 형식 요약 저장"""
        summary_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(results),
                "total_time": total_time,
                "success_rate": sum(1 for r in results if r.success) / len(results) * 100
            },
            "results": [asdict(result) for result in results],
            "statistics": self._calculate_statistics(results)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    def _save_text_summary(self, filepath: Path, results: List[TestResult], total_time: float) -> None:
        """텍스트 형식 요약 저장"""
        stats = self._calculate_statistics(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TETRIS 자동화 테스트 요약 보고서 v3.0\n")
            f.write("=" * 80 + "\n")
            f.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 실행 시간: {total_time:.2f}초\n")
            f.write(f"총 테스트 수: {len(results)}개\n")
            f.write(f"성공한 테스트: {stats['total_success']}개\n")
            f.write(f"전체 성공률: {stats['success_rate']:.1f}%\n\n")
            
            # 이미지별 통계
            f.write("=" * 40 + "[ 이미지별 통계 ]" + "=" * 40 + "\n")
            for img_name, img_stats in stats['by_image'].items():
                f.write(f"\n📷 {img_name}\n")
                f.write(f"  성공률: {img_stats['success_rate']:.1f}% "
                       f"({img_stats['success']}/{img_stats['total']})\n")
                if img_stats['avg_time'] > 0:
                    f.write(f"  평균 시간: {img_stats['avg_time']:.2f}초\n")
                    f.write(f"  시간 범위: {img_stats['min_time']:.2f}s ~ "
                           f"{img_stats['max_time']:.2f}s\n")
            
            # 성능 통계
            if stats['execution_times']:
                f.write(f"\n" + "=" * 40 + "[ 성능 통계 ]" + "=" * 40 + "\n")
                f.write(f"평균 실행 시간: {stats['avg_execution_time']:.2f}초\n")
                f.write(f"최단 실행 시간: {min(stats['execution_times']):.2f}초\n")
                f.write(f"최장 실행 시간: {max(stats['execution_times']):.2f}초\n")
                f.write(f"총 처리 시간: {sum(stats['execution_times']):.2f}초\n")
    
    def _calculate_statistics(self, results: List[TestResult]) -> Dict[str, Any]:
        """통계 계산"""
        successful_results = [r for r in results if r.success]
        execution_times = [r.execution_time for r in successful_results]
        
        # 이미지별 통계
        by_image = {}
        for result in results:
            img_name = result.image_name
            if img_name not in by_image:
                by_image[img_name] = {
                    'total': 0, 'success': 0, 'times': []
                }
            
            by_image[img_name]['total'] += 1
            if result.success:
                by_image[img_name]['success'] += 1
                by_image[img_name]['times'].append(result.execution_time)
        
        # 이미지별 통계 계산
        for img_stats in by_image.values():
            img_stats['success_rate'] = (img_stats['success'] / img_stats['total']) * 100
            if img_stats['times']:
                img_stats['avg_time'] = sum(img_stats['times']) / len(img_stats['times'])
                img_stats['min_time'] = min(img_stats['times'])
                img_stats['max_time'] = max(img_stats['times'])
            else:
                img_stats['avg_time'] = img_stats['min_time'] = img_stats['max_time'] = 0
        
        return {
            'total_success': len(successful_results),
            'success_rate': (len(successful_results) / len(results)) * 100,
            'execution_times': execution_times,
            'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'by_image': by_image
        }
    
    def print_summary(self, results: List[TestResult], total_time: float) -> None:
        """콘솔에 요약 출력"""
        stats = self._calculate_statistics(results)
        
        print("\n" + "=" * 80)
        print("🎯 테스트 완료!")
        print("=" * 80)
        print(f"⏱️  총 실행 시간: {total_time:.2f}초")
        print(f"📊 총 테스트 수: {len(results)}개")
        print(f"✅ 성공한 테스트: {stats['total_success']}개")
        print(f"📈 전체 성공률: {stats['success_rate']:.1f}%")
        
        if stats['execution_times']:
            print(f"⚡ 평균 실행 시간: {stats['avg_execution_time']:.2f}초")
        
        print(f"📁 결과 저장 위치: {self.results_path}")


# ==================== CLI Interface ====================
def create_parser() -> argparse.ArgumentParser:
    """명령행 인터페이스 파서 생성"""
    parser = argparse.ArgumentParser(
        description="TETRIS 자동화 테스트 시스템 v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  %(prog)s --test-env                    # 환경 테스트
  %(prog)s --image 0 --trials 3         # 특정 이미지 테스트
  %(prog)s --trials 5 --parallel 2      # 병렬 테스트
  %(prog)s --quick                      # 빠른 테스트 (1회)
        """
    )
    
    # 기본 옵션
    parser.add_argument('--people-count', type=int, default=1,
                       help='인원 수 (기본값: 1)')
    parser.add_argument('--trials', type=int, default=5,
                       help='시행 횟수 (기본값: 5)')
    parser.add_argument('--image', type=str,
                       help='특정 이미지만 테스트 (파일명 without extension)')
    
    # 성능 옵션
    parser.add_argument('--parallel', type=int, default=1,
                       help='병렬 워커 수 (기본값: 1, 순차 실행)')
    parser.add_argument('--timeout', type=float, default=120.0,
                       help='개별 테스트 타임아웃 (초, 기본값: 120)')
    
    # 출력 옵션
    parser.add_argument('--no-individual', action='store_true',
                       help='개별 결과 파일 저장 비활성화')
    parser.add_argument('--no-summary', action='store_true',
                       help='요약 보고서 저장 비활성화')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 출력 모드')
    
    # 특수 모드
    parser.add_argument('--test-env', action='store_true',
                       help='환경 테스트만 실행')
    parser.add_argument('--quick', action='store_true',
                       help='빠른 테스트 (첫 번째 이미지, 1회)')
    
    return parser


def test_environment():
    """환경 테스트 실행"""
    print("=" * 60)
    print("🔧 환경 테스트 실행 중...")
    print("=" * 60)
    
    try:
        config = TestConfig(trials=1, verbose=True)
        engine = TetrisTestEngine(config)
        
        # 이미지 파일 확인
        image_files = engine.get_image_files()
        if not image_files:
            print("❌ 테스트할 이미지가 없습니다.")
            return False
        
        print(f"✅ 발견된 이미지: {len(image_files)}개")
        print(f"🖼️  첫 번째 이미지: {image_files[0].name}")
        
        # 단일 테스트 실행
        print("\n🚀 단일 테스트 실행 중...")
        result = engine.execute_single_test(image_files[0], 1)
        
        if result.success:
            print(f"✅ 테스트 성공! (실행시간: {result.execution_time:.2f}초)")
            print("🎉 환경이 정상적으로 설정되었습니다.")
            return True
        else:
            print(f"❌ 테스트 실패: {result.error}")
            return False
            
    except Exception as e:
        print(f"❌ 환경 테스트 실패: {e}")
        return False


def main():
    """메인 실행 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 환경 테스트
    if args.test_env:
        success = test_environment()
        sys.exit(0 if success else 1)
    
    try:
        # 설정 생성
        config = TestConfig(
            people_count=args.people_count,
            trials=1 if args.quick else args.trials,
            timeout=args.timeout,
            parallel_workers=args.parallel,
            save_individual_results=not args.no_individual,
            save_summary=not args.no_summary,
            verbose=args.verbose
        )
        
        # 테스트 엔진 초기화
        engine = TetrisTestEngine(config)
        results_manager = ResultsManager(engine.paths['results'], args.verbose)
        
        # 이미지 파일 선택
        all_images = engine.get_image_files()
        if not all_images:
            engine.logger.error("테스트할 이미지가 없습니다.")
            sys.exit(1)
        
        if args.image:
            image_files = [img for img in all_images if img.stem == args.image]
            if not image_files:
                engine.logger.error(f"이미지 '{args.image}'를 찾을 수 없습니다.")
                print("사용 가능한 이미지:")
                for img in all_images[:10]:
                    print(f"  {img.stem}")
                sys.exit(1)
        elif args.quick:
            image_files = all_images[:1]
        else:
            image_files = all_images
        
        # 테스트 실행
        engine.logger.info(f"테스트 시작: {len(image_files)}개 이미지, {config.trials}회 시행")
        total_start = perf_counter()
        
        if config.parallel_workers > 1:
            results = engine.run_parallel_tests(image_files)
        else:
            results = engine.run_sequential_tests(image_files)
        
        total_time = perf_counter() - total_start
        
        # 이미지별 중간 보고서 추가 생성 (병렬 모드에서 누락된 경우 대비)
        if config.save_summary:
            engine._generate_missing_intermediate_summaries(results)
        
        # 결과 저장 - 개별 파일은 이미 저장되었으므로 요약만 저장
        if config.save_individual_results:
            # 누락된 파일이 있는지 확인하고 복구
            results_manager.save_individual_results(results)
        
        if config.save_summary:
            results_manager.save_summary_report(results, total_time)
        
        # 요약 출력
        results_manager.print_summary(results, total_time)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()