import {
  useState,
  useRef,
  useEffect,
  useMemo,
  forwardRef,
  useImperativeHandle,
} from 'react';
import Fuse from 'fuse.js';
import cities from '@/data/cities.json';
import { cn } from '@/lib/utils';

export interface CityComboboxHandle {
  focus: () => void;
}

interface CityComboboxProps {
  /** Controlled value — free text always accepted */
  value: string;
  /** Called when the user selects a suggestion or types free text */
  onChange: (value: string) => void;
  /** Shown below the input when truthy */
  error?: string;
  disabled?: boolean;
  id: string;
  name: string;
  label: string;
  required?: boolean;
}

const fuse = new Fuse(cities as string[], {
  threshold: 0.4,
  distance: 100,
  minMatchCharLength: 1,
});

const MAX_SUGGESTIONS = 10;

export const CityCombobox = forwardRef<CityComboboxHandle, CityComboboxProps>(
  function CityCombobox(
    { value, onChange, error, disabled, id, name, label, required },
    ref,
  ) {
    const [isOpen, setIsOpen] = useState(false);
    const [activeIndex, setActiveIndex] = useState(-1);
    const inputRef = useRef<HTMLInputElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    useImperativeHandle(ref, () => ({
      focus: () => inputRef.current?.focus(),
    }));

    const suggestions = useMemo<string[]>(() => {
      if (!value.trim()) return [];
      return fuse
        .search(value.trim())
        .slice(0, MAX_SUGGESTIONS)
        .map((r) => r.item);
    }, [value]);

    // Close dropdown on outside click
    useEffect(() => {
      function handleClickOutside(e: MouseEvent) {
        if (
          containerRef.current &&
          !containerRef.current.contains(e.target as Node)
        ) {
          setIsOpen(false);
        }
      }
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Reset active index whenever suggestions change
    useEffect(() => {
      setActiveIndex(-1);
    }, [suggestions]);

    function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
      if (!isOpen || suggestions.length === 0) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIndex((i) => Math.min(i + 1, suggestions.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === 'Enter' && activeIndex >= 0) {
        e.preventDefault();
        handleSelect(suggestions[activeIndex]);
      } else if (e.key === 'Escape') {
        setIsOpen(false);
        setActiveIndex(-1);
      }
    }

    function handleSelect(city: string) {
      onChange(city);
      setIsOpen(false);
      setActiveIndex(-1);
      inputRef.current?.focus();
    }

    function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
      onChange(e.target.value);
      setIsOpen(true);
    }

    function handleInputFocus() {
      if (value.trim()) setIsOpen(true);
    }

    const showDropdown = isOpen && suggestions.length > 0;

    return (
      <div ref={containerRef} className="relative">
        <label
          htmlFor={id}
          className="block text-sm font-medium text-gray-700"
        >
          {label}
          {required && <span className="ml-1 text-red-500" aria-hidden="true">*</span>}
        </label>

        <input
          ref={inputRef}
          type="text"
          id={id}
          name={name}
          value={value}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          autoComplete="off"
          aria-autocomplete="list"
          aria-expanded={showDropdown}
          aria-haspopup="listbox"
          aria-controls={showDropdown ? `${id}-listbox` : undefined}
          aria-activedescendant={
            activeIndex >= 0 ? `${id}-option-${activeIndex}` : undefined
          }
          className={cn(
            'mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm',
            'focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500',
            'transition-colors duration-150',
            error
              ? 'border-red-400 bg-red-50 text-red-900 placeholder-red-400 focus:border-red-500 focus:ring-red-500'
              : 'border-gray-300 bg-white text-gray-900 placeholder-gray-400',
            disabled && 'cursor-not-allowed bg-gray-100 text-gray-500',
          )}
          placeholder="Search your city or type free text…"
        />

        {/* Dropdown */}
        {showDropdown && (
          <ul
            id={`${id}-listbox`}
            role="listbox"
            aria-label="City suggestions"
            className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md border border-gray-200 bg-white py-1 shadow-lg shadow-gray-200/80"
          >
            {suggestions.map((city, i) => (
              <li
                key={city}
                id={`${id}-option-${i}`}
                role="option"
                aria-selected={i === activeIndex}
                onMouseDown={(e) => e.preventDefault()} // prevent blur before select
                onClick={() => handleSelect(city)}
                className={cn(
                  'cursor-pointer px-3 py-2 text-sm',
                  'transition-colors duration-75',
                  i === activeIndex
                    ? 'bg-indigo-50 text-indigo-700 font-medium'
                    : 'text-gray-700 hover:bg-gray-50',
                )}
              >
                {city}
              </li>
            ))}
            <li className="border-t border-gray-100 px-3 py-1.5 text-xs text-gray-400">
              Free text always accepted — type to continue
            </li>
          </ul>
        )}

        {error && (
          <p role="alert" className="mt-1 text-sm text-red-600">
            {error}
          </p>
        )}
      </div>
    );
  },
);

CityCombobox.displayName = 'CityCombobox';
