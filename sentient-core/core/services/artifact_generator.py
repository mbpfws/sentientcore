"""Artifact Generation System for Multi-Agent RAG System.

This module provides comprehensive artifact generation capabilities including:
- PDF/Markdown export
- Code generation and templates
- UI/Design artifacts (wireframes, diagrams)
- Specialized documents (business plans, marketing plans)
- Mini-app previews and code snippets
"""

import asyncio
import json
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
from jinja2 import Environment, FileSystemLoader, Template
from markdown import markdown
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.flowables import Image


class ArtifactType(str, Enum):
    """Supported artifact types."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    WIREFRAME = "wireframe"
    DIAGRAM = "diagram"
    BUSINESS_PLAN = "business_plan"
    MARKETING_PLAN = "marketing_plan"
    MINI_APP = "mini_app"
    TEMPLATE = "template"
    REPORT = "report"
    PRESENTATION = "presentation"


class ArtifactFormat(str, Enum):
    """Output formats for artifacts."""
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "md"
    JSON = "json"
    YAML = "yaml"
    SVG = "svg"
    PNG = "png"
    ZIP = "zip"


class ArtifactMetadata(BaseModel):
    """Metadata for generated artifacts."""
    title: str
    description: Optional[str] = None
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = "1.0"
    tags: List[str] = Field(default_factory=list)
    template_used: Optional[str] = None
    generation_context: Dict[str, Any] = Field(default_factory=dict)


class ArtifactRequest(BaseModel):
    """Request for artifact generation."""
    artifact_type: ArtifactType
    output_format: ArtifactFormat
    metadata: ArtifactMetadata
    content: Dict[str, Any]
    template_name: Optional[str] = None
    custom_template: Optional[str] = None
    output_path: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class ArtifactResponse(BaseModel):
    """Response from artifact generation."""
    success: bool
    artifact_id: str
    file_path: Optional[str] = None
    content: Optional[str] = None
    metadata: ArtifactMetadata
    generation_time: float
    error_message: Optional[str] = None
    preview_url: Optional[str] = None


class ArtifactGenerator(ABC):
    """Abstract base class for artifact generators."""
    
    @abstractmethod
    async def generate(self, request: ArtifactRequest) -> ArtifactResponse:
        """Generate an artifact based on the request."""
        pass
    
    @abstractmethod
    def supports_format(self, format_type: ArtifactFormat) -> bool:
        """Check if this generator supports the given format."""
        pass


class PDFGenerator(ArtifactGenerator):
    """Generator for PDF artifacts."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom PDF styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkgreen
        ))
    
    async def generate(self, request: ArtifactRequest) -> ArtifactResponse:
        """Generate PDF artifact."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create temporary file if no output path specified
            if not request.output_path:
                temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                output_path = temp_file.name
                temp_file.close()
            else:
                output_path = request.output_path
            
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            # Add title
            if request.metadata.title:
                title = Paragraph(request.metadata.title, self.styles['CustomTitle'])
                story.append(title)
                story.append(Spacer(1, 12))
            
            # Add content based on type
            if request.artifact_type == ArtifactType.BUSINESS_PLAN:
                story.extend(self._generate_business_plan_content(request.content))
            elif request.artifact_type == ArtifactType.REPORT:
                story.extend(self._generate_report_content(request.content))
            else:
                story.extend(self._generate_generic_content(request.content))
            
            # Build PDF
            doc.build(story)
            
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return ArtifactResponse(
                success=True,
                artifact_id=f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=output_path,
                metadata=request.metadata,
                generation_time=generation_time
            )
            
        except Exception as e:
            generation_time = asyncio.get_event_loop().time() - start_time
            return ArtifactResponse(
                success=False,
                artifact_id=f"pdf_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metadata=request.metadata,
                generation_time=generation_time,
                error_message=str(e)
            )
    
    def _generate_business_plan_content(self, content: Dict[str, Any]) -> List:
        """Generate business plan specific content."""
        story = []
        
        sections = [
            ('Executive Summary', content.get('executive_summary', '')),
            ('Market Analysis', content.get('market_analysis', '')),
            ('Product/Service Description', content.get('product_description', '')),
            ('Marketing Strategy', content.get('marketing_strategy', '')),
            ('Financial Projections', content.get('financial_projections', '')),
            ('Risk Analysis', content.get('risk_analysis', ''))
        ]
        
        for section_title, section_content in sections:
            if section_content:
                story.append(Paragraph(section_title, self.styles['CustomHeading']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(str(section_content), self.styles['Normal']))
                story.append(Spacer(1, 12))
        
        return story
    
    def _generate_report_content(self, content: Dict[str, Any]) -> List:
        """Generate report specific content."""
        story = []
        
        # Add summary if available
        if 'summary' in content:
            story.append(Paragraph('Summary', self.styles['CustomHeading']))
            story.append(Paragraph(content['summary'], self.styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Add sections
        if 'sections' in content:
            for section in content['sections']:
                story.append(Paragraph(section.get('title', 'Section'), self.styles['CustomHeading']))
                story.append(Paragraph(section.get('content', ''), self.styles['Normal']))
                story.append(Spacer(1, 12))
        
        return story
    
    def _generate_generic_content(self, content: Dict[str, Any]) -> List:
        """Generate generic content."""
        story = []
        
        for key, value in content.items():
            if isinstance(value, str) and value.strip():
                story.append(Paragraph(key.replace('_', ' ').title(), self.styles['CustomHeading']))
                story.append(Paragraph(value, self.styles['Normal']))
                story.append(Spacer(1, 12))
        
        return story
    
    def supports_format(self, format_type: ArtifactFormat) -> bool:
        """Check if PDF format is supported."""
        return format_type == ArtifactFormat.PDF


class MarkdownGenerator(ArtifactGenerator):
    """Generator for Markdown artifacts."""
    
    def __init__(self):
        self.template_env = Environment(
            loader=FileSystemLoader('templates'),
            autoescape=False
        )
    
    async def generate(self, request: ArtifactRequest) -> ArtifactResponse:
        """Generate Markdown artifact."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate markdown content
            if request.template_name:
                content = await self._generate_from_template(request)
            else:
                content = self._generate_markdown_content(request)
            
            # Save to file if output path specified
            if request.output_path:
                async with aiofiles.open(request.output_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
                file_path = request.output_path
            else:
                file_path = None
            
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return ArtifactResponse(
                success=True,
                artifact_id=f"md_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=file_path,
                content=content,
                metadata=request.metadata,
                generation_time=generation_time
            )
            
        except Exception as e:
            generation_time = asyncio.get_event_loop().time() - start_time
            return ArtifactResponse(
                success=False,
                artifact_id=f"md_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metadata=request.metadata,
                generation_time=generation_time,
                error_message=str(e)
            )
    
    async def _generate_from_template(self, request: ArtifactRequest) -> str:
        """Generate content from template."""
        try:
            template = self.template_env.get_template(f"{request.template_name}.md")
            return template.render(**request.content, metadata=request.metadata)
        except Exception:
            # Fallback to custom template or generic generation
            if request.custom_template:
                template = Template(request.custom_template)
                return template.render(**request.content, metadata=request.metadata)
            else:
                return self._generate_markdown_content(request)
    
    def _generate_markdown_content(self, request: ArtifactRequest) -> str:
        """Generate generic markdown content."""
        lines = []
        
        # Add title
        if request.metadata.title:
            lines.append(f"# {request.metadata.title}")
            lines.append("")
        
        # Add metadata
        if request.metadata.description:
            lines.append(f"**Description:** {request.metadata.description}")
            lines.append("")
        
        if request.metadata.author:
            lines.append(f"**Author:** {request.metadata.author}")
            lines.append("")
        
        lines.append(f"**Created:** {request.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Add content
        if request.artifact_type == ArtifactType.BUSINESS_PLAN:
            lines.extend(self._generate_business_plan_markdown(request.content))
        elif request.artifact_type == ArtifactType.CODE:
            lines.extend(self._generate_code_markdown(request.content))
        else:
            lines.extend(self._generate_generic_markdown(request.content))
        
        return "\n".join(lines)
    
    def _generate_business_plan_markdown(self, content: Dict[str, Any]) -> List[str]:
        """Generate business plan markdown."""
        lines = []
        
        sections = [
            ('## Executive Summary', content.get('executive_summary', '')),
            ('## Market Analysis', content.get('market_analysis', '')),
            ('## Product/Service Description', content.get('product_description', '')),
            ('## Marketing Strategy', content.get('marketing_strategy', '')),
            ('## Financial Projections', content.get('financial_projections', '')),
            ('## Risk Analysis', content.get('risk_analysis', ''))
        ]
        
        for section_title, section_content in sections:
            if section_content:
                lines.append(section_title)
                lines.append("")
                lines.append(str(section_content))
                lines.append("")
        
        return lines
    
    def _generate_code_markdown(self, content: Dict[str, Any]) -> List[str]:
        """Generate code documentation markdown."""
        lines = []
        
        if 'description' in content:
            lines.append("## Description")
            lines.append("")
            lines.append(content['description'])
            lines.append("")
        
        if 'code' in content:
            language = content.get('language', 'python')
            lines.append("## Code")
            lines.append("")
            lines.append(f"```{language}")
            lines.append(content['code'])
            lines.append("```")
            lines.append("")
        
        if 'usage' in content:
            lines.append("## Usage")
            lines.append("")
            lines.append(content['usage'])
            lines.append("")
        
        return lines
    
    def _generate_generic_markdown(self, content: Dict[str, Any]) -> List[str]:
        """Generate generic markdown content."""
        lines = []
        
        for key, value in content.items():
            if isinstance(value, str) and value.strip():
                lines.append(f"## {key.replace('_', ' ').title()}")
                lines.append("")
                lines.append(value)
                lines.append("")
            elif isinstance(value, (list, dict)):
                lines.append(f"## {key.replace('_', ' ').title()}")
                lines.append("")
                lines.append(f"```json")
                lines.append(json.dumps(value, indent=2))
                lines.append("```")
                lines.append("")
        
        return lines
    
    def supports_format(self, format_type: ArtifactFormat) -> bool:
        """Check if Markdown format is supported."""
        return format_type in [ArtifactFormat.MARKDOWN, ArtifactFormat.HTML]


class CodeGenerator(ArtifactGenerator):
    """Generator for code artifacts and templates."""
    
    def __init__(self):
        self.template_env = Environment(
            loader=FileSystemLoader('templates/code'),
            autoescape=False
        )
    
    async def generate(self, request: ArtifactRequest) -> ArtifactResponse:
        """Generate code artifact."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate code content
            if request.template_name:
                content = await self._generate_from_template(request)
            else:
                content = self._generate_code_content(request)
            
            # Save to file if output path specified
            if request.output_path:
                async with aiofiles.open(request.output_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
                file_path = request.output_path
            else:
                file_path = None
            
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return ArtifactResponse(
                success=True,
                artifact_id=f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=file_path,
                content=content,
                metadata=request.metadata,
                generation_time=generation_time
            )
            
        except Exception as e:
            generation_time = asyncio.get_event_loop().time() - start_time
            return ArtifactResponse(
                success=False,
                artifact_id=f"code_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metadata=request.metadata,
                generation_time=generation_time,
                error_message=str(e)
            )
    
    async def _generate_from_template(self, request: ArtifactRequest) -> str:
        """Generate code from template."""
        try:
            template = self.template_env.get_template(f"{request.template_name}.j2")
            return template.render(**request.content, metadata=request.metadata)
        except Exception:
            if request.custom_template:
                template = Template(request.custom_template)
                return template.render(**request.content, metadata=request.metadata)
            else:
                return self._generate_code_content(request)
    
    def _generate_code_content(self, request: ArtifactRequest) -> str:
        """Generate code content based on specifications."""
        content = request.content
        language = content.get('language', 'python')
        
        if language == 'python':
            return self._generate_python_code(content)
        elif language == 'javascript':
            return self._generate_javascript_code(content)
        elif language == 'typescript':
            return self._generate_typescript_code(content)
        else:
            return content.get('code', '# Generated code\npass')
    
    def _generate_python_code(self, content: Dict[str, Any]) -> str:
        """Generate Python code."""
        lines = []
        
        # Add docstring
        if content.get('description'):
            lines.append(f'"""\n{content["description"]}\n"""')
            lines.append('')
        
        # Add imports
        if content.get('imports'):
            for imp in content['imports']:
                lines.append(imp)
            lines.append('')
        
        # Add classes
        if content.get('classes'):
            for cls in content['classes']:
                lines.extend(self._generate_python_class(cls))
                lines.append('')
        
        # Add functions
        if content.get('functions'):
            for func in content['functions']:
                lines.extend(self._generate_python_function(func))
                lines.append('')
        
        # Add main code
        if content.get('main_code'):
            lines.append('if __name__ == "__main__":')
            for line in content['main_code'].split('\n'):
                lines.append(f'    {line}')
        
        return '\n'.join(lines)
    
    def _generate_python_class(self, cls_spec: Dict[str, Any]) -> List[str]:
        """Generate Python class code."""
        lines = []
        
        class_name = cls_spec.get('name', 'GeneratedClass')
        base_classes = cls_spec.get('base_classes', [])
        
        if base_classes:
            lines.append(f'class {class_name}({', '.join(base_classes)}):')
        else:
            lines.append(f'class {class_name}:')
        
        if cls_spec.get('docstring'):
            lines.append(f'    """\n    {cls_spec["docstring"]}\n    """')
        
        # Add methods
        if cls_spec.get('methods'):
            for method in cls_spec['methods']:
                method_lines = self._generate_python_function(method, indent='    ')
                lines.extend(method_lines)
        else:
            lines.append('    pass')
        
        return lines
    
    def _generate_python_function(self, func_spec: Dict[str, Any], indent: str = '') -> List[str]:
        """Generate Python function code."""
        lines = []
        
        func_name = func_spec.get('name', 'generated_function')
        params = func_spec.get('parameters', [])
        return_type = func_spec.get('return_type', '')
        
        # Build function signature
        param_str = ', '.join(params) if params else ''
        if return_type:
            signature = f'{indent}def {func_name}({param_str}) -> {return_type}:'
        else:
            signature = f'{indent}def {func_name}({param_str}):'
        
        lines.append(signature)
        
        # Add docstring
        if func_spec.get('docstring'):
            lines.append(f'{indent}    """\n{indent}    {func_spec["docstring"]}\n{indent}    """')
        
        # Add function body
        if func_spec.get('body'):
            for line in func_spec['body'].split('\n'):
                lines.append(f'{indent}    {line}')
        else:
            lines.append(f'{indent}    pass')
        
        return lines
    
    def _generate_javascript_code(self, content: Dict[str, Any]) -> str:
        """Generate JavaScript code."""
        lines = []
        
        # Add imports
        if content.get('imports'):
            for imp in content['imports']:
                lines.append(imp)
            lines.append('')
        
        # Add functions
        if content.get('functions'):
            for func in content['functions']:
                lines.extend(self._generate_javascript_function(func))
                lines.append('')
        
        # Add classes
        if content.get('classes'):
            for cls in content['classes']:
                lines.extend(self._generate_javascript_class(cls))
                lines.append('')
        
        # Add exports
        if content.get('exports'):
            lines.append('module.exports = {')
            for export in content['exports']:
                lines.append(f'  {export},')
            lines.append('};')
        
        return '\n'.join(lines)
    
    def _generate_javascript_function(self, func_spec: Dict[str, Any]) -> List[str]:
        """Generate JavaScript function code."""
        lines = []
        
        func_name = func_spec.get('name', 'generatedFunction')
        params = func_spec.get('parameters', [])
        is_async = func_spec.get('async', False)
        
        # Build function signature
        param_str = ', '.join(params) if params else ''
        if is_async:
            signature = f'async function {func_name}({param_str}) {{'
        else:
            signature = f'function {func_name}({param_str}) {{'
        
        lines.append(signature)
        
        # Add function body
        if func_spec.get('body'):
            for line in func_spec['body'].split('\n'):
                lines.append(f'  {line}')
        else:
            lines.append('  // TODO: Implement function body')
        
        lines.append('}')
        
        return lines
    
    def _generate_javascript_class(self, cls_spec: Dict[str, Any]) -> List[str]:
        """Generate JavaScript class code."""
        lines = []
        
        class_name = cls_spec.get('name', 'GeneratedClass')
        extends = cls_spec.get('extends', '')
        
        if extends:
            lines.append(f'class {class_name} extends {extends} {{')
        else:
            lines.append(f'class {class_name} {{')
        
        # Add constructor
        if cls_spec.get('constructor'):
            constructor = cls_spec['constructor']
            params = constructor.get('parameters', [])
            param_str = ', '.join(params) if params else ''
            lines.append(f'  constructor({param_str}) {{')
            
            if constructor.get('body'):
                for line in constructor['body'].split('\n'):
                    lines.append(f'    {line}')
            
            lines.append('  }')
            lines.append('')
        
        # Add methods
        if cls_spec.get('methods'):
            for method in cls_spec['methods']:
                method_lines = self._generate_javascript_function(method)
                # Indent method lines
                indented_lines = [f'  {line}' for line in method_lines]
                lines.extend(indented_lines)
                lines.append('')
        
        lines.append('}')
        
        return lines
    
    def _generate_typescript_code(self, content: Dict[str, Any]) -> str:
        """Generate TypeScript code."""
        # For now, use JavaScript generation with type annotations
        js_code = self._generate_javascript_code(content)
        
        # Add TypeScript-specific imports if needed
        if content.get('types'):
            type_imports = []
            for type_import in content['types']:
                type_imports.append(type_import)
            
            return '\n'.join(type_imports + [''] + js_code.split('\n'))
        
        return js_code
    
    def supports_format(self, format_type: ArtifactFormat) -> bool:
        """Check if code formats are supported."""
        return format_type in [ArtifactFormat.JSON, ArtifactFormat.YAML]


class ArtifactGenerationService:
    """Main service for artifact generation."""
    
    def __init__(self):
        self.generators = {
            ArtifactType.PDF: PDFGenerator(),
            ArtifactType.MARKDOWN: MarkdownGenerator(),
            ArtifactType.HTML: MarkdownGenerator(),
            ArtifactType.CODE: CodeGenerator(),
            ArtifactType.TEMPLATE: CodeGenerator(),
            ArtifactType.BUSINESS_PLAN: PDFGenerator(),
            ArtifactType.MARKETING_PLAN: PDFGenerator(),
            ArtifactType.REPORT: PDFGenerator(),
        }
        
        self.artifact_history: List[ArtifactResponse] = []
    
    async def generate_artifact(self, request: ArtifactRequest) -> ArtifactResponse:
        """Generate an artifact based on the request."""
        generator = self.generators.get(request.artifact_type)
        
        if not generator:
            return ArtifactResponse(
                success=False,
                artifact_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metadata=request.metadata,
                generation_time=0.0,
                error_message=f"No generator available for artifact type: {request.artifact_type}"
            )
        
        if not generator.supports_format(request.output_format):
            return ArtifactResponse(
                success=False,
                artifact_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metadata=request.metadata,
                generation_time=0.0,
                error_message=f"Generator does not support format: {request.output_format}"
            )
        
        response = await generator.generate(request)
        self.artifact_history.append(response)
        
        return response
    
    async def generate_mini_app_preview(self, app_spec: Dict[str, Any]) -> ArtifactResponse:
        """Generate a mini-app preview."""
        # Create HTML preview for mini-app
        html_content = self._generate_mini_app_html(app_spec)
        
        # Create temporary HTML file
        temp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        temp_file.write(html_content.encode('utf-8'))
        temp_file.close()
        
        metadata = ArtifactMetadata(
            title=app_spec.get('title', 'Mini App Preview'),
            description=app_spec.get('description', 'Generated mini-app preview'),
            author='Artifact Generator'
        )
        
        return ArtifactResponse(
            success=True,
            artifact_id=f"miniapp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            file_path=temp_file.name,
            content=html_content,
            metadata=metadata,
            generation_time=0.1,
            preview_url=f"file://{temp_file.name}"
        )
    
    def _generate_mini_app_html(self, app_spec: Dict[str, Any]) -> str:
        """Generate HTML for mini-app preview."""
        title = app_spec.get('title', 'Mini App')
        description = app_spec.get('description', '')
        components = app_spec.get('components', [])
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .component {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .button {{
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .input {{
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """
        
        # Add components
        for component in components:
            comp_type = component.get('type', 'text')
            comp_content = component.get('content', '')
            
            if comp_type == 'button':
                html += f'<div class="component"><button class="button">{comp_content}</button></div>'
            elif comp_type == 'input':
                html += f'<div class="component"><input class="input" placeholder="{comp_content}" /></div>'
            elif comp_type == 'text':
                html += f'<div class="component"><p>{comp_content}</p></div>'
            elif comp_type == 'heading':
                html += f'<div class="component"><h2>{comp_content}</h2></div>'
        
        html += """
    </div>
    <script>
        // Add basic interactivity
        document.querySelectorAll('.button').forEach(button => {
            button.addEventListener('click', () => {
                alert('Button clicked: ' + button.textContent);
            });
        });
    </script>
</body>
</html>
        """
        
        return html
    
    def get_artifact_history(self) -> List[ArtifactResponse]:
        """Get the history of generated artifacts."""
        return self.artifact_history
    
    def get_supported_types(self) -> List[ArtifactType]:
        """Get list of supported artifact types."""
        return list(self.generators.keys())
    
    def get_supported_formats(self, artifact_type: ArtifactType) -> List[ArtifactFormat]:
        """Get supported formats for a specific artifact type."""
        generator = self.generators.get(artifact_type)
        if not generator:
            return []
        
        supported_formats = []
        for format_type in ArtifactFormat:
            if generator.supports_format(format_type):
                supported_formats.append(format_type)
        
        return supported_formats


# Global service instance
artifact_service = ArtifactGenerationService()


# Convenience functions
async def generate_pdf_report(title: str, content: Dict[str, Any], output_path: Optional[str] = None) -> ArtifactResponse:
    """Generate a PDF report."""
    metadata = ArtifactMetadata(title=title, description="Generated PDF report")
    request = ArtifactRequest(
        artifact_type=ArtifactType.REPORT,
        output_format=ArtifactFormat.PDF,
        metadata=metadata,
        content=content,
        output_path=output_path
    )
    return await artifact_service.generate_artifact(request)


async def generate_markdown_documentation(title: str, content: Dict[str, Any], output_path: Optional[str] = None) -> ArtifactResponse:
    """Generate Markdown documentation."""
    metadata = ArtifactMetadata(title=title, description="Generated Markdown documentation")
    request = ArtifactRequest(
        artifact_type=ArtifactType.MARKDOWN,
        output_format=ArtifactFormat.MARKDOWN,
        metadata=metadata,
        content=content,
        output_path=output_path
    )
    return await artifact_service.generate_artifact(request)


async def generate_code_template(language: str, template_spec: Dict[str, Any], output_path: Optional[str] = None) -> ArtifactResponse:
    """Generate code template."""
    metadata = ArtifactMetadata(title=f"{language.title()} Code Template", description=f"Generated {language} code")
    content = {"language": language, **template_spec}
    request = ArtifactRequest(
        artifact_type=ArtifactType.CODE,
        output_format=ArtifactFormat.JSON,
        metadata=metadata,
        content=content,
        output_path=output_path
    )
    return await artifact_service.generate_artifact(request)


async def generate_business_plan(plan_data: Dict[str, Any], output_path: Optional[str] = None) -> ArtifactResponse:
    """Generate business plan PDF."""
    metadata = ArtifactMetadata(title="Business Plan", description="Generated business plan document")
    request = ArtifactRequest(
        artifact_type=ArtifactType.BUSINESS_PLAN,
        output_format=ArtifactFormat.PDF,
        metadata=metadata,
        content=plan_data,
        output_path=output_path
    )
    return await artifact_service.generate_artifact(request)