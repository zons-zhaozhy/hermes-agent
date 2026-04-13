"""Integration tests for output filter across all tool types.

Verifies that filtering is correctly integrated into terminal_tool.py,
code_execution_tool.py, browser_tool.py, web_tools.py, and file_tools.py.
"""

import json
import os
import sys
import tempfile
from unittest.mock import patch

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


class TestTerminalToolIntegration:
    """Verify terminal_tool.py integration."""

    def test_terminal_tool_calls_filter(self):
        """Check that terminal_tool imports and calls filter_terminal_output."""
        from tools.terminal_tool import terminal_tool
        
        # Read the file to verify import
        with open(os.path.join(PROJECT_ROOT, 'tools', 'terminal_tool.py'), 'r') as f:
            content = f.read()
        
        assert 'from tools.output_filter import filter_terminal_output' in content
        assert 'output = filter_terminal_output(output, command=command)' in content
        
        print("✓ terminal_tool.py integration verified")


class TestCodeExecutionToolIntegration:
    """Verify code_execution_tool.py integration."""

    def test_code_execution_tool_calls_filter(self):
        """Check that code_execution_tool imports and calls filter_code_execution_output."""
        # Read the file to verify import
        with open(os.path.join(PROJECT_ROOT, 'tools', 'code_execution_tool.py'), 'r') as f:
            content = f.read()
        
        assert 'from tools.output_filter import filter_code_execution_output' in content
        assert 'filter_code_execution_output' in content
        
        print("✓ code_execution_tool.py integration verified")


class TestBrowserToolIntegration:
    """Verify browser_tool.py integration."""

    def test_browser_snapshot_calls_filter(self):
        """Check that browser_snapshot imports and calls filter_browser_output."""
        # Read the file to verify import
        with open(os.path.join(PROJECT_ROOT, 'tools', 'browser_tool.py'), 'r') as f:
            content = f.read()
        
        assert 'from tools.output_filter import filter_browser_output' in content
        assert 'snapshot_text = filter_browser_output(snapshot_text, browser_action="snapshot")' in content
        assert 'analysis = filter_browser_output(analysis, browser_action="vision")' in content
        
        print("✓ browser_tool.py integration verified")


class TestWebExtractToolIntegration:
    """Verify web_tools.py integration."""

    def test_web_extract_tool_calls_filter(self):
        """Check that web_tools.py imports and calls filter_web_extract_output."""
        with open(os.path.join(PROJECT_ROOT, 'tools', 'web_tools.py'), 'r') as f:
            content = f.read()

        assert 'from tools.output_filter import filter_web_extract_output' in content
        assert 'filter_web_extract_output(cleaned_result' in content

        print("✓ web_tools.py integration verified")


class TestSearchToolIntegration:
    """Verify file_tools.py integration."""

    def test_search_tool_calls_filter(self):
        """Check that file_tools.py imports and calls filter_search_output."""
        with open(os.path.join(PROJECT_ROOT, 'tools', 'file_tools.py'), 'r') as f:
            content = f.read()

        assert 'from tools.output_filter import filter_search_output' in content
        assert 'filter_search_output(result_json' in content

        print("✓ file_tools.py integration verified")


class TestConfiguration:
    """Verify configuration system works."""

    def test_config_yaml_parsing(self):
        """Test that config.yaml parsing works."""
        from tools.output_filter import _get_filter_config
        
        # Test with env var enabled
        with patch.dict(os.environ, {'HERMES_OUTPUT_FILTER': 'true'}):
            config = _get_filter_config()
            assert config['enabled'] is True
            assert config['level'] == 'moderate'
            assert 'terminal' in config['tools']
            assert 'code_execution' in config['tools']
            assert 'browser' in config['tools']
            assert 'web_extract' in config['tools']
            assert 'search' in config['tools']
        
        # Test with env var disabled
        with patch.dict(os.environ, {'HERMES_OUTPUT_FILTER': 'false'}):
            config = _get_filter_config()
            assert config['enabled'] is False
        
        print("✓ Configuration system verified")


class TestEndToEnd:
    """End-to-end verification of token savings."""

    def test_realistic_scenario(self):
        """Simulate a realistic development session with multiple tools."""
        import os
        os.environ['HERMES_OUTPUT_FILTER'] = 'true'
        
        from tools.output_filter import (
            filter_terminal_output,
            filter_code_execution_output,
            filter_browser_output,
        )
        
        # 1. Terminal: cargo test output
        cargo_output = []
        cargo_output.append('Compiling myproject v0.1.0')
        cargo_output.append('[========================================] 100%')
        for i in range(80):
            cargo_output.append(f'test test_{i:03d} ... ok')
        cargo_output.append('test test_edge ... FAILED')
        cargo_output.append('warning: unused import: `std::collections::HashMap`')
        cargo_output.append('')
        cargo_output.append('test result: FAILED. 80 passed; 1 failed;')
        
        terminal_filtered = filter_terminal_output('\n'.join(cargo_output), command='cargo test')
        terminal_savings = 1 - len(terminal_filtered) / len('\n'.join(cargo_output))
        assert terminal_savings > 0.5, f"Expected >50% savings, got {terminal_savings:.1%}"
        print(f"✓ Terminal: {terminal_savings:.1%} savings")
        
        # 2. Code execution: Python script with verbose repeated output
        script_output = []
        script_output.append('Starting data processing...')
        for i in range(50):
            script_output.append(f'Processing item {i}: success')
            script_output.append(f'  Detail: row={i} col={i} checksum=0x{hash(i) & 0xFFFF:04x}')
        # Many duplicate warnings
        for _ in range(20):
            script_output.append('warning: deprecated API used')
        for _ in range(15):
            script_output.append('DeprecationWarning: old_function is deprecated')
        script_output.append('Processing complete.')
        
        code_filtered = filter_code_execution_output('\n'.join(script_output), script_info='data processing')
        code_savings = 1 - len(code_filtered) / len('\n'.join(script_output))
        assert code_savings > 0.2, f"Expected >20% savings, got {code_savings:.1%}"
        print(f"✓ Code execution: {code_savings:.1%} savings")
        
        # 3. Browser: Vision analysis (should have minimal savings - it's already clean)
        vision_output = '''The page contains a form with username and password fields.
There is a submit button labeled "Login".
The page uses a blue color scheme.
No errors are visible.'''
        
        browser_filtered = filter_browser_output(vision_output, browser_action='vision')
        # Clean text should pass through unchanged
        assert browser_filtered == vision_output
        print("✓ Browser: clean text passed through unchanged")
        
        print("✓ All end-to-end tests passed")


def run_all_tests():
    """Run all integration tests."""
    print("Running output filter integration tests...")
    print("=" * 60)
    
    test_classes = [
        TestTerminalToolIntegration(),
        TestCodeExecutionToolIntegration(),
        TestBrowserToolIntegration(),
        TestWebExtractToolIntegration(),
        TestSearchToolIntegration(),
        TestConfiguration(),
        TestEndToEnd(),
    ]
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")
        print("-" * 40)
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                if callable(method):
                    try:
                        method()
                        print(f"  ✓ {method_name}")
                    except Exception as e:
                        print(f"  ✗ {method_name}: {e}")
                        raise
    
    print("\n" + "=" * 60)
    print("All integration tests passed! ✓")


if __name__ == '__main__':
    run_all_tests()
