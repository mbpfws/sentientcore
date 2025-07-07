"""Comprehensive testing framework for Sentient Core."""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestCategory(str, Enum):
    """Test categories."""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    HEALTH = "health"
    API = "api"


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    category: TestCategory
    status: TestStatus
    duration: float = 0.0
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Test suite containing multiple tests."""
    name: str
    tests: List['BaseTest'] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    parallel: bool = False


class BaseTest(ABC):
    """Base class for all tests."""
    
    def __init__(self, name: str, category: TestCategory, timeout: float = 30.0):
        self.name = name
        self.category = category
        self.timeout = timeout
        self.logger = logging.getLogger(f"test.{name}")
    
    @abstractmethod
    async def run(self) -> TestResult:
        """Execute the test."""
        pass
    
    async def setup(self):
        """Setup before test execution."""
        pass
    
    async def teardown(self):
        """Cleanup after test execution."""
        pass


class FunctionTest(BaseTest):
    """Test that wraps a function."""
    
    def __init__(
        self,
        name: str,
        test_func: Callable,
        category: TestCategory = TestCategory.UNIT,
        timeout: float = 30.0,
        setup_func: Optional[Callable] = None,
        teardown_func: Optional[Callable] = None
    ):
        super().__init__(name, category, timeout)
        self.test_func = test_func
        self.setup_func = setup_func
        self.teardown_func = teardown_func
    
    async def setup(self):
        if self.setup_func:
            if asyncio.iscoroutinefunction(self.setup_func):
                await self.setup_func()
            else:
                self.setup_func()
    
    async def teardown(self):
        if self.teardown_func:
            if asyncio.iscoroutinefunction(self.teardown_func):
                await self.teardown_func()
            else:
                self.teardown_func()
    
    async def run(self) -> TestResult:
        start_time = time.time()
        
        try:
            await self.setup()
            
            if asyncio.iscoroutinefunction(self.test_func):
                result = await asyncio.wait_for(self.test_func(), timeout=self.timeout)
            else:
                result = self.test_func()
            
            await self.teardown()
            
            duration = time.time() - start_time
            
            return TestResult(
                name=self.name,
                category=self.category,
                status=TestStatus.PASSED,
                duration=duration,
                metadata={"result": result}
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return TestResult(
                name=self.name,
                category=self.category,
                status=TestStatus.ERROR,
                duration=duration,
                error_message=f"Test timed out after {self.timeout} seconds"
            )
            
        except AssertionError as e:
            duration = time.time() - start_time
            return TestResult(
                name=self.name,
                category=self.category,
                status=TestStatus.FAILED,
                duration=duration,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=self.name,
                category=self.category,
                status=TestStatus.ERROR,
                duration=duration,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        
        finally:
            try:
                await self.teardown()
            except Exception as e:
                self.logger.error(f"Error in teardown: {e}")


class ServiceHealthTest(BaseTest):
    """Test for service health checks."""
    
    def __init__(self, service_name: str, health_check_func: Callable):
        super().__init__(f"health_check_{service_name}", TestCategory.HEALTH)
        self.service_name = service_name
        self.health_check_func = health_check_func
    
    async def run(self) -> TestResult:
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(self.health_check_func):
                is_healthy = await self.health_check_func()
            else:
                is_healthy = self.health_check_func()
            
            duration = time.time() - start_time
            
            if is_healthy:
                return TestResult(
                    name=self.name,
                    category=self.category,
                    status=TestStatus.PASSED,
                    duration=duration,
                    metadata={"service": self.service_name, "healthy": True}
                )
            else:
                return TestResult(
                    name=self.name,
                    category=self.category,
                    status=TestStatus.FAILED,
                    duration=duration,
                    error_message=f"Service {self.service_name} is not healthy",
                    metadata={"service": self.service_name, "healthy": False}
                )
                
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=self.name,
                category=self.category,
                status=TestStatus.ERROR,
                duration=duration,
                error_message=str(e),
                traceback=traceback.format_exc(),
                metadata={"service": self.service_name}
            )


class PerformanceTest(BaseTest):
    """Test for performance benchmarks."""
    
    def __init__(
        self,
        name: str,
        test_func: Callable,
        max_duration: float,
        iterations: int = 1
    ):
        super().__init__(name, TestCategory.PERFORMANCE)
        self.test_func = test_func
        self.max_duration = max_duration
        self.iterations = iterations
    
    async def run(self) -> TestResult:
        durations = []
        
        try:
            for i in range(self.iterations):
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(self.test_func):
                    await self.test_func()
                else:
                    self.test_func()
                
                duration = time.time() - start_time
                durations.append(duration)
            
            avg_duration = sum(durations) / len(durations)
            max_duration_actual = max(durations)
            
            if max_duration_actual <= self.max_duration:
                status = TestStatus.PASSED
                error_message = None
            else:
                status = TestStatus.FAILED
                error_message = f"Performance test failed: {max_duration_actual:.3f}s > {self.max_duration:.3f}s"
            
            return TestResult(
                name=self.name,
                category=self.category,
                status=status,
                duration=avg_duration,
                error_message=error_message,
                metadata={
                    "iterations": self.iterations,
                    "avg_duration": avg_duration,
                    "max_duration": max_duration_actual,
                    "threshold": self.max_duration,
                    "all_durations": durations
                }
            )
            
        except Exception as e:
            return TestResult(
                name=self.name,
                category=self.category,
                status=TestStatus.ERROR,
                duration=sum(durations) / len(durations) if durations else 0,
                error_message=str(e),
                traceback=traceback.format_exc(),
                metadata={"completed_iterations": len(durations)}
            )


class TestRunner:
    """Test execution engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_suites: Dict[str, TestSuite] = {}
        self.results: List[TestResult] = []
    
    def add_test_suite(self, suite: TestSuite):
        """Add a test suite."""
        self.test_suites[suite.name] = suite
        self.logger.info(f"Added test suite: {suite.name} with {len(suite.tests)} tests")
    
    def add_test(self, suite_name: str, test: BaseTest):
        """Add a test to a suite."""
        if suite_name not in self.test_suites:
            self.test_suites[suite_name] = TestSuite(name=suite_name)
        
        self.test_suites[suite_name].tests.append(test)
        self.logger.info(f"Added test {test.name} to suite {suite_name}")
    
    async def run_suite(self, suite_name: str) -> List[TestResult]:
        """Run a specific test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite {suite_name} not found")
        
        suite = self.test_suites[suite_name]
        self.logger.info(f"Running test suite: {suite_name}")
        
        # Run setup
        if suite.setup_func:
            try:
                if asyncio.iscoroutinefunction(suite.setup_func):
                    await suite.setup_func()
                else:
                    suite.setup_func()
            except Exception as e:
                self.logger.error(f"Suite setup failed: {e}")
                return []
        
        # Run tests
        suite_results = []
        
        if suite.parallel:
            # Run tests in parallel
            tasks = [test.run() for test in suite.tests]
            suite_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            for i, result in enumerate(suite_results):
                if isinstance(result, Exception):
                    suite_results[i] = TestResult(
                        name=suite.tests[i].name,
                        category=suite.tests[i].category,
                        status=TestStatus.ERROR,
                        error_message=str(result),
                        traceback=traceback.format_exc()
                    )
        else:
            # Run tests sequentially
            for test in suite.tests:
                try:
                    result = await test.run()
                    suite_results.append(result)
                except Exception as e:
                    suite_results.append(TestResult(
                        name=test.name,
                        category=test.category,
                        status=TestStatus.ERROR,
                        error_message=str(e),
                        traceback=traceback.format_exc()
                    ))
        
        # Run teardown
        if suite.teardown_func:
            try:
                if asyncio.iscoroutinefunction(suite.teardown_func):
                    await suite.teardown_func()
                else:
                    suite.teardown_func()
            except Exception as e:
                self.logger.error(f"Suite teardown failed: {e}")
        
        self.results.extend(suite_results)
        
        # Log results
        passed = len([r for r in suite_results if r.status == TestStatus.PASSED])
        failed = len([r for r in suite_results if r.status == TestStatus.FAILED])
        errors = len([r for r in suite_results if r.status == TestStatus.ERROR])
        
        self.logger.info(f"Suite {suite_name} completed: {passed} passed, {failed} failed, {errors} errors")
        
        return suite_results
    
    async def run_all_suites(self) -> Dict[str, List[TestResult]]:
        """Run all test suites."""
        all_results = {}
        
        for suite_name in self.test_suites:
            all_results[suite_name] = await self.run_suite(suite_name)
        
        return all_results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get test execution summary."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "errors": 0, "success_rate": 0}
        
        total = len(self.results)
        passed = len([r for r in self.results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.results if r.status == TestStatus.FAILED])
        errors = len([r for r in self.results if r.status == TestStatus.ERROR])
        skipped = len([r for r in self.results if r.status == TestStatus.SKIPPED])
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "success_rate": round(success_rate, 2),
            "avg_duration": round(sum(r.duration for r in self.results) / total, 3) if total > 0 else 0
        }
    
    def get_failed_tests(self) -> List[TestResult]:
        """Get all failed tests."""
        return [r for r in self.results if r.status in [TestStatus.FAILED, TestStatus.ERROR]]
    
    def clear_results(self):
        """Clear test results."""
        self.results.clear()


# Global test runner instance
test_runner = TestRunner()


# Utility functions for creating tests
def create_function_test(
    name: str,
    test_func: Callable,
    category: TestCategory = TestCategory.UNIT,
    timeout: float = 30.0
) -> FunctionTest:
    """Create a function test."""
    return FunctionTest(name, test_func, category, timeout)


def create_health_test(service_name: str, health_check_func: Callable) -> ServiceHealthTest:
    """Create a service health test."""
    return ServiceHealthTest(service_name, health_check_func)


def create_performance_test(
    name: str,
    test_func: Callable,
    max_duration: float,
    iterations: int = 1
) -> PerformanceTest:
    """Create a performance test."""
    return PerformanceTest(name, test_func, max_duration, iterations)