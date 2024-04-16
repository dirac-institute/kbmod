class BatchSearchManager:
    def __init__(self, stack_search, search_list, min_observations):
        """Manages a batch search over a list of Trajectory instances using a StackSearch instance.

        Parameters
        ----------
        stack_search: `StackSearch`
            StackSearch instance to use for the batch search.
        search_list: `list[Trajectory]`
            List of Trajectory instances to search along the stack.
        min_observations: `int`
            Minimum number of observations required to consider a candidate.
        """
        self.stack_search = stack_search
        self.search_list = search_list
        self.min_observations = min_observations

    def __enter__(self):
        # Prepare memory for the batch search.
        self.stack_search.prepare_search(self.search_list, self.min_observations)
        return self.stack_search

    def __exit__(self, *_):
        """
        This method is called when exiting the context.
        It cleans up resources by calling `finish_search` on the StackSearch instance.
        We return False to indicate that we do not want to suppress any exceptions that may have been raised.
        """
        # Clean up
        self.stack_search.finish_search()
        return False
