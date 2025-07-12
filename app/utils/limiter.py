from functools import wraps
import time
from flask import request, jsonify
from typing import Dict, List
import threading

# Thread-safe rate limiting with cleanup
class RateLimiter:
    def __init__(self):
        self.request_counts: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # Clean up every 5 minutes
        self.last_cleanup = time.time()
    
    def _cleanup_old_requests(self, current_time: float, window: int):
        """Clean up old requests to prevent memory leaks"""
        if current_time - self.last_cleanup > self.cleanup_interval:
            with self.lock:
                for client_ip in list(self.request_counts.keys()):
                    self.request_counts[client_ip] = [
                        req_time for req_time in self.request_counts[client_ip]
                        if current_time - req_time < window
                    ]
                    # Remove empty entries
                    if not self.request_counts[client_ip]:
                        del self.request_counts[client_ip]
                self.last_cleanup = current_time
    
    def is_allowed(self, client_ip: str, max_requests: int, window: int) -> bool:
        """Check if request is allowed based on rate limits"""
        current_time = time.time()
        
        # Periodic cleanup
        self._cleanup_old_requests(current_time, window)
        
        with self.lock:
            if client_ip not in self.request_counts:
                self.request_counts[client_ip] = []
            
            # Clean old requests for this client
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if current_time - req_time < window
            ]
            
            # Check if under limit
            if len(self.request_counts[client_ip]) >= max_requests:
                return False
            
            # Add current request
            self.request_counts[client_ip].append(current_time)
            return True
    
    def get_remaining_requests(self, client_ip: str, max_requests: int, window: int) -> int:
        """Get remaining requests for a client"""
        current_time = time.time()
        
        with self.lock:
            if client_ip not in self.request_counts:
                return max_requests
            
            # Clean old requests
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if current_time - req_time < window
            ]
            
            return max(0, max_requests - len(self.request_counts[client_ip]))
    
    def get_reset_time(self, client_ip: str, window: int) -> float:
        """Get time when rate limit resets for a client"""
        current_time = time.time()
        
        with self.lock:
            if client_ip not in self.request_counts or not self.request_counts[client_ip]:
                return current_time
            
            oldest_request = min(self.request_counts[client_ip])
            return oldest_request + window

# Global rate limiter instance
_rate_limiter = RateLimiter()

def rate_limit(max_requests=100, window=3600, per_ip=True):
    """
    Rate limiting decorator
    
    Args:
        max_requests: Maximum number of requests allowed
        window: Time window in seconds
        per_ip: Whether to apply rate limiting per IP address
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if per_ip:
                client_ip = request.remote_addr or 'unknown'
            else:
                client_ip = 'global'
            
            if not _rate_limiter.is_allowed(client_ip, max_requests, window):
                remaining = _rate_limiter.get_remaining_requests(client_ip, max_requests, window)
                reset_time = _rate_limiter.get_reset_time(client_ip, window)
                
                response = jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Limit: {max_requests} per {window} seconds',
                    'remaining': remaining,
                    'reset_time': reset_time
                })
                response.status_code = 429
                response.headers['X-RateLimit-Limit'] = str(max_requests)
                response.headers['X-RateLimit-Remaining'] = str(remaining)
                response.headers['X-RateLimit-Reset'] = str(int(reset_time))
                return response
            
            # Add rate limit headers to successful responses
            remaining = _rate_limiter.get_remaining_requests(client_ip, max_requests, window)
            reset_time = _rate_limiter.get_reset_time(client_ip, window)
            
            response = f(*args, **kwargs)
            
            # Add headers if response is a Flask response object
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit'] = str(max_requests)
                response.headers['X-RateLimit-Remaining'] = str(remaining)
                response.headers['X-RateLimit-Reset'] = str(int(reset_time))
            
            return response
        return decorated_function
    return decorator

# Convenience decorators for common use cases
def api_rate_limit(max_requests=60, window=3600):
    """Standard API rate limiting: 60 requests per hour"""
    return rate_limit(max_requests=max_requests, window=window)

def upload_rate_limit(max_requests=10, window=3600):
    """Rate limiting for file uploads: 10 uploads per hour"""
    return rate_limit(max_requests=max_requests, window=window)

def analysis_rate_limit(max_requests=20, window=3600):
    """Rate limiting for analysis endpoints: 20 analyses per hour"""
    return rate_limit(max_requests=max_requests, window=window)

def strict_rate_limit(max_requests=5, window=600):
    """Strict rate limiting: 5 requests per 10 minutes"""
    return rate_limit(max_requests=max_requests, window=window)

# Global rate limiter functions
def get_rate_limit_status(client_ip: str, max_requests: int, window: int) -> Dict:
    """Get current rate limit status for a client"""
    remaining = _rate_limiter.get_remaining_requests(client_ip, max_requests, window)
    reset_time = _rate_limiter.get_reset_time(client_ip, window)
    
    return {
        'limit': max_requests,
        'remaining': remaining,
        'reset_time': reset_time,
        'window': window
    }

def reset_rate_limit(client_ip: str):
    """Reset rate limit for a specific client (admin function)"""
    with _rate_limiter.lock:
        if client_ip in _rate_limiter.request_counts:
            del _rate_limiter.request_counts[client_ip]

def get_all_rate_limits() -> Dict:
    """Get all current rate limits (admin function)"""
    with _rate_limiter.lock:
        return {
            ip: len(requests) 
            for ip, requests in _rate_limiter.request_counts.items()
        }