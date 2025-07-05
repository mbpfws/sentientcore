"""
The Front-End Developer Agent
Handles all tasks related to UI/UX design and front-end implementation.
"""

from core.models import AppState, EnhancedTask, AgentType, LogEntry
from core.services.llm_service import EnhancedLLMService
from core.agents.base_agent import BaseAgent, AgentCapability, ActivityType
from typing import Dict, Any, Optional, List
import json
import asyncio

class FrontendDeveloperAgent(BaseAgent):
    """
    The Front-End Developer Agent is responsible for:
    - Creating wireframes and mockups.
    - Developing interactive prototypes.
    - Building the final user interface using frameworks like Next.js or Streamlit.
    - Collaborating with the backend developer to integrate APIs.
    """

    def __init__(self, llm_service: EnhancedLLMService, agent_id: str = "frontend_developer_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Frontend Developer Agent",
            capabilities=[AgentCapability.FRONTEND_DEVELOPMENT, AgentCapability.UI_DESIGN, AgentCapability.PROTOTYPING],
            description="Specialized agent for frontend development, UI/UX design, and user interface implementation"
        )
        self.llm_service = llm_service
        self.supported_frameworks = ['react', 'nextjs', 'vue', 'angular', 'html', 'css', 'javascript']
        self.ui_components = []
        self.design_patterns = []

    def can_handle_task(self, task: EnhancedTask) -> bool:
        """
        Determines if this agent can handle the given task.
        """
        frontend_keywords = [
            'frontend', 'front-end', 'ui', 'ux', 'interface', 'design', 'wireframe', 
            'mockup', 'prototype', 'react', 'vue', 'angular', 'html', 'css', 'javascript',
            'component', 'layout', 'responsive', 'styling', 'user interface'
        ]
        task_description = task.description.lower()
        return any(keyword in task_description for keyword in frontend_keywords)
    
    async def process_task(self, task: EnhancedTask, state: Optional[AppState] = None) -> Dict:
        """
        Processes a frontend development task.
        """
        try:
            self.log_activity(ActivityType.TASK_STARTED, f"Processing frontend task: {task.description}")
            
            # Determine the type of frontend task
            task_type = self._determine_task_type(task.description)
            
            result = {}
            
            if task_type == 'design':
                result = await self._handle_design_task(task)
            elif task_type == 'component':
                result = await self._handle_component_task(task)
            elif task_type == 'layout':
                result = await self._handle_layout_task(task)
            elif task_type == 'styling':
                result = await self._handle_styling_task(task)
            elif task_type == 'prototype':
                result = await self._handle_prototype_task(task)
            else:
                result = await self._handle_general_frontend_task(task)
            
            self.log_activity(ActivityType.TASK_COMPLETED, f"Frontend task completed: {task.description}")
            
            return {
                "status": "completed",
                "task_type": task_type,
                "result": result
            }
            
        except Exception as e:
            self.handle_error(e, f"Error processing frontend task: {task.description}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _determine_task_type(self, description: str) -> str:
        """
        Determines the type of frontend task based on description.
        """
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['design', 'wireframe', 'mockup', 'ux']):
            return 'design'
        elif any(word in description_lower for word in ['component', 'widget', 'element']):
            return 'component'
        elif any(word in description_lower for word in ['layout', 'grid', 'structure']):
            return 'layout'
        elif any(word in description_lower for word in ['style', 'css', 'theme', 'color']):
            return 'styling'
        elif any(word in description_lower for word in ['prototype', 'interactive', 'demo']):
            return 'prototype'
        else:
            return 'general'
    
    async def _handle_design_task(self, task: EnhancedTask) -> Dict:
        """
        Handles UI/UX design tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Creating UI/UX design")
        
        try:
            # Generate design specifications using LLM
            design_prompt = f"""
            Create a comprehensive UI/UX design specification for: {task.description}
            
            Include:
            1. User interface layout
            2. Component hierarchy
            3. Color scheme and typography
            4. User interaction patterns
            5. Responsive design considerations
            
            Provide the response in JSON format with clear sections.
            """
            
            response = await self.llm_service.generate_response(
                prompt=design_prompt,
                model="compound-beta"
            )
            
            design_spec = self._parse_json_response(response)
            
            return {
                "design_specification": design_spec,
                "wireframes": self._generate_wireframe_structure(design_spec),
                "design_tokens": self._extract_design_tokens(design_spec)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in design task: {e}")
            raise
    
    async def _handle_component_task(self, task: EnhancedTask) -> Dict:
        """
        Handles component development tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Developing UI component")
        
        try:
            component_prompt = f"""
            Create a reusable UI component for: {task.description}
            
            Generate:
            1. Component structure (React/Vue/Angular)
            2. Props/properties interface
            3. Styling (CSS/SCSS)
            4. Usage examples
            5. Accessibility considerations
            
            Provide code examples and documentation.
            """
            
            response = await self.llm_service.generate_response(
                prompt=component_prompt,
                model="compound-beta"
            )
            
            return {
                "component_code": response,
                "component_type": self._identify_component_type(task.description),
                "dependencies": self._extract_dependencies(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in component task: {e}")
            raise
    
    async def _handle_layout_task(self, task: EnhancedTask) -> Dict:
        """
        Handles layout and structure tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Creating layout structure")
        
        try:
            layout_prompt = f"""
            Create a responsive layout structure for: {task.description}
            
            Include:
            1. HTML structure
            2. CSS Grid/Flexbox layout
            3. Responsive breakpoints
            4. Semantic HTML elements
            5. Accessibility features
            
            Provide complete code with comments.
            """
            
            response = await self.llm_service.generate_response(
                prompt=layout_prompt,
                model="compound-beta"
            )
            
            return {
                "layout_code": response,
                "layout_type": self._identify_layout_type(task.description),
                "responsive_features": self._extract_responsive_features(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in layout task: {e}")
            raise
    
    async def _handle_styling_task(self, task: EnhancedTask) -> Dict:
        """
        Handles styling and theming tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Creating styles and themes")
        
        try:
            styling_prompt = f"""
            Create comprehensive styling for: {task.description}
            
            Include:
            1. CSS/SCSS styles
            2. Color palette
            3. Typography system
            4. Spacing and layout utilities
            5. Animation and transitions
            
            Provide organized CSS with variables and mixins.
            """
            
            response = await self.llm_service.generate_response(
                prompt=styling_prompt,
                model="compound-beta"
            )
            
            return {
                "styles": response,
                "theme_tokens": self._extract_theme_tokens(response),
                "css_methodology": self._identify_css_methodology(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in styling task: {e}")
            raise
    
    async def _handle_prototype_task(self, task: EnhancedTask) -> Dict:
        """
        Handles interactive prototype tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Creating interactive prototype")
        
        try:
            prototype_prompt = f"""
            Create an interactive prototype for: {task.description}
            
            Include:
            1. Interactive elements
            2. User flow implementation
            3. State management
            4. Event handlers
            5. Animation and feedback
            
            Provide functional code with interaction logic.
            """
            
            response = await self.llm_service.generate_response(
                prompt=prototype_prompt,
                model="compound-beta"
            )
            
            return {
                "prototype_code": response,
                "interactions": self._extract_interactions(response),
                "user_flows": self._identify_user_flows(task.description)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in prototype task: {e}")
            raise
    
    async def _handle_general_frontend_task(self, task: EnhancedTask) -> Dict:
        """
        Handles general frontend development tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Processing general frontend task")
        
        try:
            general_prompt = f"""
            Implement frontend solution for: {task.description}
            
            Provide:
            1. Complete implementation
            2. Best practices
            3. Performance considerations
            4. Testing approach
            5. Documentation
            
            Use modern frontend technologies and patterns.
            """
            
            response = await self.llm_service.generate_response(
                prompt=general_prompt,
                model="compound-beta"
            )
            
            return {
                "implementation": response,
                "technologies_used": self._identify_technologies(response),
                "best_practices": self._extract_best_practices(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in general frontend task: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict:
        """
        Safely parses JSON response from LLM.
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON from markdown or other formatting
            import re
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return {"raw_response": response}
    
    def _generate_wireframe_structure(self, design_spec: Dict) -> Dict:
        """
        Generates wireframe structure from design specification.
        """
        return {
            "layout_sections": design_spec.get("layout", {}),
            "component_hierarchy": design_spec.get("components", []),
            "navigation_structure": design_spec.get("navigation", {})
        }
    
    def _extract_design_tokens(self, design_spec: Dict) -> Dict:
        """
        Extracts design tokens from specification.
        """
        return {
            "colors": design_spec.get("colors", {}),
            "typography": design_spec.get("typography", {}),
            "spacing": design_spec.get("spacing", {}),
            "shadows": design_spec.get("shadows", {})
        }
    
    def _identify_component_type(self, description: str) -> str:
        """
        Identifies the type of component being developed.
        """
        description_lower = description.lower()
        
        if 'button' in description_lower:
            return 'button'
        elif 'form' in description_lower:
            return 'form'
        elif 'modal' in description_lower:
            return 'modal'
        elif 'card' in description_lower:
            return 'card'
        elif 'navigation' in description_lower or 'nav' in description_lower:
            return 'navigation'
        else:
            return 'custom'
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """
        Extracts dependencies from generated code.
        """
        dependencies = []
        import re
        
        # Extract import statements
        import_matches = re.findall(r'import.*?from [\'"]([^\'"]+)[\'"]', code)
        dependencies.extend(import_matches)
        
        # Extract require statements
        require_matches = re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', code)
        dependencies.extend(require_matches)
        
        return list(set(dependencies))
    
    def _identify_layout_type(self, description: str) -> str:
        """
        Identifies the type of layout being created.
        """
        description_lower = description.lower()
        
        if 'grid' in description_lower:
            return 'grid'
        elif 'flex' in description_lower:
            return 'flexbox'
        elif 'sidebar' in description_lower:
            return 'sidebar'
        elif 'header' in description_lower:
            return 'header'
        elif 'footer' in description_lower:
            return 'footer'
        else:
            return 'custom'
    
    def _extract_responsive_features(self, code: str) -> List[str]:
        """
        Extracts responsive design features from code.
        """
        features = []
        
        if '@media' in code:
            features.append('media_queries')
        if 'grid' in code.lower():
            features.append('css_grid')
        if 'flex' in code.lower():
            features.append('flexbox')
        if 'viewport' in code.lower():
            features.append('viewport_meta')
        
        return features
    
    def _extract_theme_tokens(self, styles: str) -> Dict:
        """
        Extracts theme tokens from CSS/SCSS.
        """
        tokens = {
            "colors": [],
            "fonts": [],
            "spacing": [],
            "breakpoints": []
        }
        
        import re
        
        # Extract CSS custom properties
        color_matches = re.findall(r'--[\w-]*color[\w-]*:\s*([^;]+)', styles)
        tokens["colors"].extend(color_matches)
        
        font_matches = re.findall(r'--[\w-]*font[\w-]*:\s*([^;]+)', styles)
        tokens["fonts"].extend(font_matches)
        
        return tokens
    
    def _identify_css_methodology(self, styles: str) -> str:
        """
        Identifies the CSS methodology used.
        """
        if 'bem' in styles.lower() or '__' in styles:
            return 'BEM'
        elif 'module' in styles.lower():
            return 'CSS_Modules'
        elif 'styled' in styles.lower():
            return 'Styled_Components'
        else:
            return 'Standard_CSS'
    
    def _extract_interactions(self, code: str) -> List[str]:
        """
        Extracts interaction patterns from prototype code.
        """
        interactions = []
        
        if 'onClick' in code or 'click' in code.lower():
            interactions.append('click')
        if 'onHover' in code or 'hover' in code.lower():
            interactions.append('hover')
        if 'onFocus' in code or 'focus' in code.lower():
            interactions.append('focus')
        if 'animation' in code.lower():
            interactions.append('animation')
        if 'transition' in code.lower():
            interactions.append('transition')
        
        return interactions
    
    def _identify_user_flows(self, description: str) -> List[str]:
        """
        Identifies user flows from task description.
        """
        flows = []
        description_lower = description.lower()
        
        if 'login' in description_lower:
            flows.append('authentication')
        if 'signup' in description_lower or 'register' in description_lower:
            flows.append('registration')
        if 'checkout' in description_lower:
            flows.append('checkout')
        if 'search' in description_lower:
            flows.append('search')
        if 'navigation' in description_lower:
            flows.append('navigation')
        
        return flows
    
    def _identify_technologies(self, code: str) -> List[str]:
        """
        Identifies technologies used in the implementation.
        """
        technologies = []
        code_lower = code.lower()
        
        for framework in self.supported_frameworks:
            if framework in code_lower:
                technologies.append(framework)
        
        return technologies
    
    def _extract_best_practices(self, code: str) -> List[str]:
        """
        Extracts best practices mentioned in the code.
        """
        practices = []
        code_lower = code.lower()
        
        if 'accessibility' in code_lower or 'aria' in code_lower:
            practices.append('accessibility')
        if 'semantic' in code_lower:
            practices.append('semantic_html')
        if 'performance' in code_lower:
            practices.append('performance_optimization')
        if 'responsive' in code_lower:
            practices.append('responsive_design')
        if 'test' in code_lower:
            practices.append('testing')
        
        return practices

    def invoke(self, state: AppState) -> AppState:
        """
        Legacy method for backward compatibility.
        The main entry point for the front-end development process.
        """
        print("---FRONT-END DEVELOPER AGENT---")
        
        self.log_activity(ActivityType.PROCESSING, "Legacy invoke method called")
        
        # Placeholder logic
        state.logs.append(LogEntry(
            source="FrontendDeveloperAgent",
            message="Front-end development logic not yet implemented."
        ))
        
        return state