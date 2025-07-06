import { useState, useCallback, useEffect, useRef } from 'react';
import { coreServicesClient } from '../api';

export interface Artifact {
  id: string;
  type: 'research' | 'plan' | 'specification' | 'code' | 'documentation' | 'analysis' | 'report';
  title: string;
  description?: string;
  content?: string;
  url?: string;
  download_url?: string;
  preview_url?: string;
  metadata: {
    created_at: Date;
    updated_at: Date;
    created_by: string; // agent or user
    workflow_id?: string;
    session_id?: string;
    file_size?: number;
    file_format?: 'md' | 'pdf' | 'json' | 'txt' | 'html' | 'docx';
    tags?: string[];
    priority?: 'low' | 'medium' | 'high' | 'critical';
    status?: 'draft' | 'review' | 'approved' | 'archived';
    version?: string;
    parent_id?: string; // For versioning
    related_artifacts?: string[];
  };
  stats?: {
    views: number;
    downloads: number;
    last_accessed: Date;
    sharing_enabled: boolean;
  };
  content_preview?: string; // First few lines for quick preview
  search_keywords?: string[];
}

export interface ArtifactCollection {
  id: string;
  name: string;
  description?: string;
  artifact_ids: string[];
  created_at: Date;
  updated_at: Date;
  metadata?: {
    workflow_id?: string;
    session_id?: string;
    tags?: string[];
    color?: string;
    icon?: string;
  };
}

export interface ArtifactFilter {
  type?: Artifact['type'][];
  created_by?: string[];
  workflow_id?: string;
  session_id?: string;
  status?: Artifact['metadata']['status'][];
  priority?: Artifact['metadata']['priority'][];
  tags?: string[];
  date_range?: {
    start: Date;
    end: Date;
  };
  search_query?: string;
  file_format?: string[];
}

export interface ArtifactManagerState {
  artifacts: Record<string, Artifact>;
  collections: Record<string, ArtifactCollection>;
  currentFilter: ArtifactFilter;
  selectedArtifacts: string[];
  currentArtifact: Artifact | null;
  isLoading: boolean;
  isUploading: boolean;
  uploadProgress: number;
  lastError: string | null;
  viewMode: 'grid' | 'list' | 'timeline';
  sortBy: 'created_at' | 'updated_at' | 'title' | 'type' | 'priority';
  sortOrder: 'asc' | 'desc';
  previewMode: boolean;
  searchResults: string[];
  recentlyViewed: string[];
  favorites: string[];
  settings: {
    autoPreview: boolean;
    enableVersioning: boolean;
    maxRecentItems: number;
    defaultFileFormat: string;
    enableAutoTags: boolean;
    compressionEnabled: boolean;
  };
}

const initialState: ArtifactManagerState = {
  artifacts: {},
  collections: {},
  currentFilter: {},
  selectedArtifacts: [],
  currentArtifact: null,
  isLoading: false,
  isUploading: false,
  uploadProgress: 0,
  lastError: null,
  viewMode: 'grid',
  sortBy: 'updated_at',
  sortOrder: 'desc',
  previewMode: false,
  searchResults: [],
  recentlyViewed: [],
  favorites: [],
  settings: {
    autoPreview: true,
    enableVersioning: true,
    maxRecentItems: 20,
    defaultFileFormat: 'md',
    enableAutoTags: true,
    compressionEnabled: false
  }
};

export function useArtifactManager() {
  const [state, setState] = useState<ArtifactManagerState>(initialState);
  const stateRef = useRef(state);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Keep ref in sync
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Load settings and favorites from localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const storedSettings = localStorage.getItem('artifact_manager_settings');
        if (storedSettings) {
          const settings = JSON.parse(storedSettings);
          setState(prev => ({ ...prev, settings: { ...prev.settings, ...settings } }));
        }
        
        const storedFavorites = localStorage.getItem('artifact_manager_favorites');
        if (storedFavorites) {
          const favorites = JSON.parse(storedFavorites);
          setState(prev => ({ ...prev, favorites }));
        }
        
        const storedRecent = localStorage.getItem('artifact_manager_recent');
        if (storedRecent) {
          const recentlyViewed = JSON.parse(storedRecent);
          setState(prev => ({ ...prev, recentlyViewed }));
        }
      } catch (error) {
        console.warn('Failed to load artifact manager data:', error);
      }
    }
  }, []);

  // Save settings to localStorage
  const saveSettings = useCallback((newSettings: Partial<ArtifactManagerState['settings']>) => {
    const updatedSettings = { ...stateRef.current.settings, ...newSettings };
    setState(prev => ({ ...prev, settings: updatedSettings }));
    
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('artifact_manager_settings', JSON.stringify(updatedSettings));
      } catch (error) {
        console.warn('Failed to save artifact manager settings:', error);
      }
    }
  }, []);

  // Load artifacts from backend
  const loadArtifacts = useCallback(async (filter?: ArtifactFilter) => {
    setState(prev => ({ ...prev, isLoading: true, lastError: null }));
    
    try {
      // Use core services to get research artifacts and other data
      const response = await coreServicesClient.listResearchArtifacts();
      
      if (response.success && response.artifacts) {
        const artifactsMap: Record<string, Artifact> = {};
        
        response.artifacts.forEach((artifact: any) => {
          const processedArtifact: Artifact = {
            id: artifact.id || `artifact_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type: artifact.type || 'research',
            title: artifact.title || artifact.name || 'Untitled Artifact',
            description: artifact.description,
            content: artifact.content,
            url: artifact.url,
            download_url: artifact.download_url,
            preview_url: artifact.preview_url,
            metadata: {
              created_at: artifact.created_at ? new Date(artifact.created_at) : new Date(),
              updated_at: artifact.updated_at ? new Date(artifact.updated_at) : new Date(),
              created_by: artifact.created_by || 'system',
              workflow_id: artifact.workflow_id,
              session_id: artifact.session_id,
              file_size: artifact.file_size,
              file_format: artifact.file_format || 'md',
              tags: artifact.tags || [],
              priority: artifact.priority || 'medium',
              status: artifact.status || 'draft',
              version: artifact.version || '1.0',
              parent_id: artifact.parent_id,
              related_artifacts: artifact.related_artifacts || []
            },
            stats: {
              views: artifact.views || 0,
              downloads: artifact.downloads || 0,
              last_accessed: artifact.last_accessed ? new Date(artifact.last_accessed) : new Date(),
              sharing_enabled: artifact.sharing_enabled || false
            },
            content_preview: artifact.content ? artifact.content.substring(0, 200) + '...' : undefined,
            search_keywords: generateSearchKeywords(artifact)
          };
          
          artifactsMap[processedArtifact.id] = processedArtifact;
        });
        
        setState(prev => ({
          ...prev,
          artifacts: artifactsMap,
          isLoading: false
        }));
        
        // Apply filter if provided
        if (filter) {
          applyFilter(filter);
        }
      } else {
        setState(prev => ({
          ...prev,
          isLoading: false,
          lastError: 'Failed to load artifacts'
        }));
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        lastError: error instanceof Error ? error.message : 'Unknown error loading artifacts'
      }));
    }
  }, []);

  // Generate search keywords for an artifact
  const generateSearchKeywords = useCallback((artifact: any): string[] => {
    const keywords = new Set<string>();
    
    // Add title words
    if (artifact.title) {
      artifact.title.toLowerCase().split(/\s+/).forEach((word: string) => {
        if (word.length > 2) keywords.add(word);
      });
    }
    
    // Add description words
    if (artifact.description) {
      artifact.description.toLowerCase().split(/\s+/).forEach((word: string) => {
        if (word.length > 2) keywords.add(word);
      });
    }
    
    // Add tags
    if (artifact.tags) {
      artifact.tags.forEach((tag: string) => keywords.add(tag.toLowerCase()));
    }
    
    // Add type
    if (artifact.type) {
      keywords.add(artifact.type.toLowerCase());
    }
    
    // Add created_by
    if (artifact.created_by) {
      keywords.add(artifact.created_by.toLowerCase());
    }
    
    return Array.from(keywords);
  }, []);

  // Create new artifact
  const createArtifact = useCallback(async (artifactData: {
    type: Artifact['type'];
    title: string;
    description?: string;
    content?: string;
    metadata?: Partial<Artifact['metadata']>;
    file?: File;
  }): Promise<string> => {
    setState(prev => ({ ...prev, isUploading: true, uploadProgress: 0, lastError: null }));
    
    try {
      const artifactId = `artifact_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      let content = artifactData.content;
      let fileSize = content ? new Blob([content]).size : 0;
      let fileFormat = stateRef.current.settings.defaultFileFormat as any;
      
      // Handle file upload
      if (artifactData.file) {
        const formData = new FormData();
        formData.append('file', artifactData.file);
        formData.append('artifact_id', artifactId);
        
        // Simulate upload progress
        const uploadInterval = setInterval(() => {
          setState(prev => ({
            ...prev,
            uploadProgress: Math.min(prev.uploadProgress + 10, 90)
          }));
        }, 100);
        
        try {
          // Here you would upload to your backend
          // const uploadResponse = await uploadFile(formData);
          
          clearInterval(uploadInterval);
          setState(prev => ({ ...prev, uploadProgress: 100 }));
          
          fileSize = artifactData.file.size;
          fileFormat = artifactData.file.name.split('.').pop() as any || 'txt';
          
          if (!content) {
            content = await artifactData.file.text();
          }
        } catch (uploadError) {
          clearInterval(uploadInterval);
          throw uploadError;
        }
      }
      
      const newArtifact: Artifact = {
        id: artifactId,
        type: artifactData.type,
        title: artifactData.title,
        description: artifactData.description,
        content,
        metadata: {
          created_at: new Date(),
          updated_at: new Date(),
          created_by: 'user',
          file_size: fileSize,
          file_format: fileFormat,
          tags: [],
          priority: 'medium',
          status: 'draft',
          version: '1.0',
          ...artifactData.metadata
        },
        stats: {
          views: 0,
          downloads: 0,
          last_accessed: new Date(),
          sharing_enabled: false
        },
        content_preview: content ? content.substring(0, 200) + '...' : undefined,
        search_keywords: generateSearchKeywords({
          title: artifactData.title,
          description: artifactData.description,
          type: artifactData.type,
          tags: artifactData.metadata?.tags || [],
          created_by: 'user'
        })
      };
      
      // Auto-generate tags if enabled
      if (stateRef.current.settings.enableAutoTags) {
        newArtifact.metadata.tags = generateAutoTags(newArtifact);
        newArtifact.search_keywords = [...(newArtifact.search_keywords || []), ...newArtifact.metadata.tags];
      }
      
      setState(prev => ({
        ...prev,
        artifacts: {
          ...prev.artifacts,
          [artifactId]: newArtifact
        },
        isUploading: false,
        uploadProgress: 0
      }));
      
      // Store in backend
      try {
        await coreServicesClient.storeMemory({
          layer: 'session',
          type: 'artifact',
          key: `artifact_${artifactId}`,
          data: newArtifact,
          metadata: {
            artifact_type: artifactData.type,
            created_by: 'user',
            timestamp: new Date().toISOString()
          }
        });
      } catch (storeError) {
        console.warn('Failed to store artifact in backend:', storeError);
      }
      
      return artifactId;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isUploading: false,
        uploadProgress: 0,
        lastError: error instanceof Error ? error.message : 'Failed to create artifact'
      }));
      throw error;
    }
  }, [generateSearchKeywords]);

  // Generate auto tags based on content analysis
  const generateAutoTags = useCallback((artifact: Artifact): string[] => {
    const tags = new Set<string>();
    
    // Add type-based tags
    tags.add(artifact.type);
    
    // Analyze content for common patterns
    const content = (artifact.content || artifact.description || '').toLowerCase();
    
    // Technical tags
    if (content.includes('api') || content.includes('endpoint')) tags.add('api');
    if (content.includes('database') || content.includes('sql')) tags.add('database');
    if (content.includes('frontend') || content.includes('ui')) tags.add('frontend');
    if (content.includes('backend') || content.includes('server')) tags.add('backend');
    if (content.includes('test') || content.includes('testing')) tags.add('testing');
    if (content.includes('deploy') || content.includes('deployment')) tags.add('deployment');
    if (content.includes('security') || content.includes('auth')) tags.add('security');
    if (content.includes('performance') || content.includes('optimization')) tags.add('performance');
    
    // Language/framework tags
    if (content.includes('react') || content.includes('jsx')) tags.add('react');
    if (content.includes('typescript') || content.includes('ts')) tags.add('typescript');
    if (content.includes('python') || content.includes('py')) tags.add('python');
    if (content.includes('javascript') || content.includes('js')) tags.add('javascript');
    if (content.includes('node') || content.includes('nodejs')) tags.add('nodejs');
    
    // Priority-based tags
    if (artifact.metadata.priority === 'critical') tags.add('urgent');
    if (artifact.metadata.priority === 'high') tags.add('important');
    
    return Array.from(tags).slice(0, 10); // Limit to 10 auto-generated tags
  }, []);

  // Update artifact
  const updateArtifact = useCallback(async (artifactId: string, updates: Partial<Artifact>) => {
    const existingArtifact = stateRef.current.artifacts[artifactId];
    if (!existingArtifact) {
      throw new Error('Artifact not found');
    }
    
    // Create new version if versioning is enabled
    let updatedArtifact = { ...existingArtifact, ...updates };
    
    if (stateRef.current.settings.enableVersioning && (updates.content || updates.title)) {
      const newVersion = incrementVersion(existingArtifact.metadata.version || '1.0');
      updatedArtifact.metadata = {
        ...updatedArtifact.metadata,
        version: newVersion,
        updated_at: new Date(),
        parent_id: existingArtifact.id
      };
      
      // Create new artifact for the version
      const versionId = `${artifactId}_v${newVersion.replace('.', '_')}`;
      updatedArtifact.id = versionId;
      
      setState(prev => ({
        ...prev,
        artifacts: {
          ...prev.artifacts,
          [versionId]: updatedArtifact,
          [artifactId]: {
            ...existingArtifact,
            metadata: {
              ...existingArtifact.metadata,
              status: 'archived'
            }
          }
        }
      }));
    } else {
      updatedArtifact.metadata = {
        ...updatedArtifact.metadata,
        updated_at: new Date()
      };
      
      setState(prev => ({
        ...prev,
        artifacts: {
          ...prev.artifacts,
          [artifactId]: updatedArtifact
        }
      }));
    }
    
    // Update search keywords if content changed
    if (updates.content || updates.title || updates.description) {
      updatedArtifact.search_keywords = generateSearchKeywords(updatedArtifact);
      updatedArtifact.content_preview = updatedArtifact.content ? 
        updatedArtifact.content.substring(0, 200) + '...' : undefined;
    }
  }, [generateSearchKeywords]);

  // Increment version string
  const incrementVersion = useCallback((version: string): string => {
    const parts = version.split('.');
    const major = parseInt(parts[0] || '1');
    const minor = parseInt(parts[1] || '0');
    const patch = parseInt(parts[2] || '0');
    
    return `${major}.${minor}.${patch + 1}`;
  }, []);

  // Delete artifact
  const deleteArtifact = useCallback((artifactId: string) => {
    setState(prev => {
      const { [artifactId]: deleted, ...remainingArtifacts } = prev.artifacts;
      return {
        ...prev,
        artifacts: remainingArtifacts,
        selectedArtifacts: prev.selectedArtifacts.filter(id => id !== artifactId),
        currentArtifact: prev.currentArtifact?.id === artifactId ? null : prev.currentArtifact
      };
    });
  }, []);

  // Apply filter
  const applyFilter = useCallback((filter: ArtifactFilter) => {
    setState(prev => ({ ...prev, currentFilter: filter }));
    
    const filtered = Object.values(stateRef.current.artifacts).filter(artifact => {
      // Type filter
      if (filter.type && filter.type.length > 0 && !filter.type.includes(artifact.type)) {
        return false;
      }
      
      // Created by filter
      if (filter.created_by && filter.created_by.length > 0 && 
          !filter.created_by.includes(artifact.metadata.created_by)) {
        return false;
      }
      
      // Status filter
      if (filter.status && filter.status.length > 0 && 
          !filter.status.includes(artifact.metadata.status)) {
        return false;
      }
      
      // Priority filter
      if (filter.priority && filter.priority.length > 0 && 
          !filter.priority.includes(artifact.metadata.priority)) {
        return false;
      }
      
      // Tags filter
      if (filter.tags && filter.tags.length > 0) {
        const hasMatchingTag = filter.tags.some(tag => 
          artifact.metadata.tags?.includes(tag)
        );
        if (!hasMatchingTag) return false;
      }
      
      // Date range filter
      if (filter.date_range) {
        const createdAt = artifact.metadata.created_at.getTime();
        if (createdAt < filter.date_range.start.getTime() || 
            createdAt > filter.date_range.end.getTime()) {
          return false;
        }
      }
      
      // Search query filter
      if (filter.search_query) {
        const query = filter.search_query.toLowerCase();
        const searchableText = [
          artifact.title,
          artifact.description,
          artifact.content,
          ...(artifact.metadata.tags || []),
          ...(artifact.search_keywords || [])
        ].join(' ').toLowerCase();
        
        if (!searchableText.includes(query)) {
          return false;
        }
      }
      
      return true;
    });
    
    setState(prev => ({
      ...prev,
      searchResults: filtered.map(a => a.id)
    }));
  }, []);

  // Search artifacts with debouncing
  const searchArtifacts = useCallback((query: string) => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    
    searchTimeoutRef.current = setTimeout(() => {
      applyFilter({ ...stateRef.current.currentFilter, search_query: query });
    }, 300);
  }, [applyFilter]);

  // Select artifact and track viewing
  const selectArtifact = useCallback((artifactId: string) => {
    const artifact = stateRef.current.artifacts[artifactId];
    if (!artifact) return;
    
    setState(prev => ({ ...prev, currentArtifact: artifact }));
    
    // Update stats
    updateArtifact(artifactId, {
      stats: {
        ...artifact.stats!,
        views: (artifact.stats?.views || 0) + 1,
        last_accessed: new Date()
      }
    });
    
    // Add to recently viewed
    const updatedRecent = [artifactId, ...stateRef.current.recentlyViewed.filter(id => id !== artifactId)]
      .slice(0, stateRef.current.settings.maxRecentItems);
    
    setState(prev => ({ ...prev, recentlyViewed: updatedRecent }));
    
    // Save to localStorage
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('artifact_manager_recent', JSON.stringify(updatedRecent));
      } catch (error) {
        console.warn('Failed to save recently viewed:', error);
      }
    }
  }, [updateArtifact]);

  // Toggle favorite
  const toggleFavorite = useCallback((artifactId: string) => {
    const isFavorite = stateRef.current.favorites.includes(artifactId);
    const updatedFavorites = isFavorite 
      ? stateRef.current.favorites.filter(id => id !== artifactId)
      : [...stateRef.current.favorites, artifactId];
    
    setState(prev => ({ ...prev, favorites: updatedFavorites }));
    
    // Save to localStorage
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('artifact_manager_favorites', JSON.stringify(updatedFavorites));
      } catch (error) {
        console.warn('Failed to save favorites:', error);
      }
    }
  }, []);

  // Get filtered and sorted artifacts
  const getFilteredArtifacts = useCallback(() => {
    let artifacts = stateRef.current.searchResults.length > 0 
      ? stateRef.current.searchResults.map(id => stateRef.current.artifacts[id]).filter(Boolean)
      : Object.values(stateRef.current.artifacts);
    
    // Sort artifacts
    artifacts.sort((a, b) => {
      const aValue = getSortValue(a, stateRef.current.sortBy);
      const bValue = getSortValue(b, stateRef.current.sortBy);
      
      if (stateRef.current.sortOrder === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });
    
    return artifacts;
  }, []);

  // Get sort value for artifact
  const getSortValue = useCallback((artifact: Artifact, sortBy: string): any => {
    switch (sortBy) {
      case 'created_at':
        return artifact.metadata.created_at.getTime();
      case 'updated_at':
        return artifact.metadata.updated_at.getTime();
      case 'title':
        return artifact.title.toLowerCase();
      case 'type':
        return artifact.type;
      case 'priority':
        const priorityOrder = { low: 1, medium: 2, high: 3, critical: 4 };
        return priorityOrder[artifact.metadata.priority || 'medium'];
      default:
        return artifact.metadata.created_at.getTime();
    }
  }, []);

  // Get statistics
  const getStats = useCallback(() => {
    const artifacts = Object.values(stateRef.current.artifacts);
    const now = Date.now();
    const last24h = artifacts.filter(a => now - a.metadata.created_at.getTime() < 24 * 60 * 60 * 1000);
    const last7d = artifacts.filter(a => now - a.metadata.created_at.getTime() < 7 * 24 * 60 * 60 * 1000);
    
    return {
      total_artifacts: artifacts.length,
      by_type: artifacts.reduce((acc, a) => {
        acc[a.type] = (acc[a.type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>),
      by_status: artifacts.reduce((acc, a) => {
        acc[a.metadata.status || 'draft'] = (acc[a.metadata.status || 'draft'] || 0) + 1;
        return acc;
      }, {} as Record<string, number>),
      created_last_24h: last24h.length,
      created_last_7d: last7d.length,
      total_views: artifacts.reduce((sum, a) => sum + (a.stats?.views || 0), 0),
      total_downloads: artifacts.reduce((sum, a) => sum + (a.stats?.downloads || 0), 0),
      favorites_count: stateRef.current.favorites.length,
      collections_count: Object.keys(stateRef.current.collections).length
    };
  }, []);

  return {
    // State
    state,
    
    // Actions
    loadArtifacts,
    createArtifact,
    updateArtifact,
    deleteArtifact,
    selectArtifact,
    toggleFavorite,
    applyFilter,
    searchArtifacts,
    saveSettings,
    
    // Computed values
    filteredArtifacts: getFilteredArtifacts(),
    stats: getStats(),
    hasArtifacts: Object.keys(state.artifacts).length > 0,
    hasSelection: state.selectedArtifacts.length > 0,
    isFiltered: Object.keys(state.currentFilter).length > 0
  };
}

export default useArtifactManager;